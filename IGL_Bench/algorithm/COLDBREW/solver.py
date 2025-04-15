import copy

from torch_geometric.utils import negative_sampling

from IGL_Bench.algorithm.COLDBREW.GNN_normalizations import TeacherGNN
from IGL_Bench.algorithm.COLDBREW.utils import toitem, linkp_loss_eva
from IGL_Bench.backbone.gcn import GCN_node_sparse
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score


class COLDBREW_node_solver:
    def __init__(self, config, dataset, device='cuda'):
        self.config = config
        self.dataset = dataset
        self.device = device

        self.model = {}
        self.optimizer = {}
        self.bag = {}
        self.initializtion()

        self.model['default'] = self.model['default'].to(device)
        self.dataset = self.dataset.to(self.device)

    def initializtion(self):
        num_classes = self.dataset.y.numpy().max().item() + 1
        self.model['default'] = TeacherGNN(self.config, None).to(self.device)

        self.optimizer['default'] = torch.optim.Adam(self.model['default'].parameters(), lr=self.config.lr,
                                                     weight_decay=self.config.weight_decay)

    def reset_parameters(self):
        """Reset model parameters and reinitialize optimizer."""
        # Reset model parameters
        for model_name, model in self.model.items():
            if hasattr(model, 'reset_parameters'):
                model.reset_parameters()
            else:
                for layer in model.modules():
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()

        # Reinitialize optimizer
        self.optimizer = {}
        for model_name, model in self.model.items():
            self.optimizer[model_name] = torch.optim.Adam(
                model.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.weight_decay
            )

    def train(self):
        self.reset_parameters()
        num_epochs = getattr(self.config, 'epoch', 500)
        patience = getattr(self.config, 'patience', 100)
        least_epoch = getattr(self.config, 'least_epoch', 40)

        criterion = torch.nn.CrossEntropyLoss()

        best_loss = float('inf')
        patience_counter = 0
        best_val_accuracy = 0
        best_model_state_dict = None

        for epoch in range(1, num_epochs + 1):
            self.model['default'].train()
            loss, linkp_train, linkp_test = -1, 0, 0
            assert self.config.has_loss_component_nodewise or self.config.has_loss_component_edgewise, 'setting no node-wise and no edge-wise loss for teacherGNN! at least set one of them!'
            res = self.model['default'].get_3_embs(self.dataset.x, self.dataset.edge_index, self.dataset.train_mask)
            raw_logits, emb4classi_full, emb4linkp = res.emb4classi, res.emb4classi_full, res.emb4linkp
            if self.config.has_loss_component_nodewise:
                # ========= classification: train =========
                logits = F.log_softmax(raw_logits, 1)
                loss_semantic = criterion(logits, self.dataset.y[self.dataset.train_mask])
                loss = loss_semantic * self.config.TeacherGNN.lossa_semantic
                if self.model['default'].se_reg_all is not None:
                    loss += self.config.se_reg * self.model['default'].se_reg_all

                result = []
                if self.config.want_headtail:
                    all_node_logits = self.model['default'].get_3_embs(self.dataset.x,
                                                                       self.dataset.edge_index).emb4classi
                    lrn_targ = self.dataset.y

                    batch_idx = self.dataset.large_deg_idx
                    train_head, test_head = self.eval_headtail__traintest_v2(all_node_logits[batch_idx],
                                                                             lrn_targ[batch_idx], batch_idx,
                                                                             self.cal_acc_rounded100)

                    batch_idx = self.dataset.small_deg_idx
                    train_tail, test_tail = self.eval_headtail__traintest_v2(all_node_logits[batch_idx],
                                                                             lrn_targ[batch_idx], batch_idx,
                                                                             self.cal_acc_rounded100)

                    head_tail_iso = [test_head, test_tail]
                    result.extend(head_tail_iso)

                    if self.config.use_special_split:
                        batch_idx = self.dataset.zero_deg_idx
                        train_iso, test_iso = self.eval_headtail__traintest_v2(all_node_logits[batch_idx],
                                                                               lrn_targ[batch_idx], batch_idx,
                                                                               self.cal_acc_rounded100)
                        result.extend([test_iso])

                self.bag['head_tail_iso'] = result

            if self.config.has_loss_component_edgewise:
                emb4linkp = res.commonEmb  # for linkp, must use full node embs (without applying train_mask!!)
                # ======= link prediction: train =======
                loss_structure, linkp_train = self.getLinkp_loss_eva(emb4linkp, 'train')
                # ======= link prediction: eva =======
                _, linkp_test = self.getLinkp_loss_eva(emb4linkp, 'test')
                if loss is None:
                    loss = loss_structure * self.config.TeacherGNN.lossa_structure
                else:
                    loss = loss + loss_structure * self.config.TeacherGNN.lossa_structure

            self.optimizer['default'].zero_grad()
            loss.backward()
            self.optimizer['default'].step()

            print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}")

            val_accuracy = self.eval(metric="accuracy")

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0
                best_model_state_dict = copy.deepcopy(self.model['default'].state_dict())
            else:
                patience_counter += 1

            if patience_counter >= patience and epoch > least_epoch:
                print(f"Early stopping at epoch {epoch + 1}.")
                break

        self.model['default'].load_state_dict(best_model_state_dict)
        print("Training Finished!")

    def eval_headtail__traintest_v2(self, emb2, lrn_targ, subsets, metricfun):
        # emb2/lrn_target: not full node emb, but a subset, whose index is given by indices
        # subsets can be either head or tail, in either indices or mask
        actual_train_mask = self.dataset.train_mask[subsets]
        on_train = torch.where(actual_train_mask)[0]
        on_test = torch.where(~actual_train_mask)[0]

        metric_train = metricfun(emb2[on_train], lrn_targ[on_train])
        metric_test = metricfun(emb2[on_test], lrn_targ[on_test])
        return metric_train, metric_test

    def cal_acc_rounded100(self, output, labels):
        # output = output.to(labels.device)
        _, indices = torch.max(output, dim=1)
        correct = torch.sum(indices == labels) / len(labels)
        return toitem(correct * 100)

    def getLinkp_loss_eva(self, emb, mode):
        h_emb, t_emb, nh_emb, nt_emb = self.gen_pn_edges(emb, mode)
        loss, eva_score = linkp_loss_eva(h_emb, t_emb, nh_emb, nt_emb)
        return loss, eva_score

    def gen_pn_edges(self, nodes_emb, mode):
        # nodes_emb must contain all nodes
        # mode: train / test
        # assert self.args.use_special_split  # otherwise not implemented
        if mode=='train':
            valid_edge_mask = self.dataset.train_mask[self.dataset.edge_index[0]] * self.dataset.train_mask[self.dataset.edge_index[1]]  # len = edge_index
            valid_edge_index = self.dataset.edge_index[:,valid_edge_mask]  # len = num of true
        elif mode=='test':
            test_mask = ~ self.dataset.train_mask
            valid_edge_mask = test_mask[self.dataset.edge_index[0]] * test_mask[self.dataset.edge_index[1]]  # len = edge_index_bkup
            valid_edge_index = self.dataset.edge_index[:,valid_edge_mask]

        else:
            raise NotImplementedError
        samp_size_p = self.config.samp_size_p

        samp_edge_p_idx = np.random.choice(valid_edge_index.shape[1], samp_size_p)
        samp_edge_p = valid_edge_index[:,samp_edge_p_idx]
        samp_edge_n = self.my_negative_sampling(mode)

        h_emb = nodes_emb[samp_edge_p[0]]
        t_emb = nodes_emb[samp_edge_p[1]]
        nh_emb = nodes_emb[samp_edge_n[0]]
        nt_emb = nodes_emb[samp_edge_n[1]]
        return h_emb, t_emb, nh_emb, nt_emb

    def my_negative_sampling(self, mode):
        # how to sample neg edge:
        # first get neg sample for all edges in the graph, then screen them according to train/test split: for training set, neg edge samples are those ori & dst nodes all falls within training split; for test set, neg edge samples are those at least one node of ori/dst falls within the test split.

        sampled_all = []
        N_sampled = 0

        if mode == 'train':
            samp_size_n = self.config.samp_size_n_train
            samp_size_n_sub = max(samp_size_n // 4, 50)  # do neg sample in small batches to prevend over flood
            while N_sampled < samp_size_n:
                edge_samp = negative_sampling(self.dataset.edge_index, num_neg_samples=samp_size_n_sub,
                                              force_undirected=True)
                fall_in_mask = self.dataset.train_mask[edge_samp[0]] * self.dataset.train_mask[edge_samp[1]]
                edge_samp = edge_samp[:, fall_in_mask]
                N_sampled += edge_samp.size(1)
                sampled_all.append(edge_samp)

        elif mode == 'test':
            samp_size_n = self.config.samp_size_p * self.config.samp_size_n_test_times_p
            samp_size_n_sub = max(samp_size_n // 4, 50)  # do neg sample in small batches to prevend over flood
            while N_sampled < samp_size_n:
                edge_samp = negative_sampling(self.dataset.edge_index, num_neg_samples=samp_size_n_sub,
                                              force_undirected=True)
                fall_in_mask = ~ (self.dataset.train_mask[edge_samp[0]] * self.dataset.train_mask[edge_samp[1]])
                edge_samp = edge_samp[:, fall_in_mask]
                N_sampled += edge_samp.size(1)
                sampled_all.append(edge_samp)

        sampled_all = torch.cat(sampled_all, dim=1)
        return sampled_all

    def eval(self, metric="accuracy"):
        """ Evaluate the model on the validation or test set using the selected metric. """
        self.model['default'].eval()
        all_labels = self.dataset.y[self.dataset.val_mask].cpu().numpy()

        with torch.no_grad():
            res = self.model['default'].get_3_embs(self.dataset.x, self.dataset.edge_index)
            raw_logits = res.emb4classi
            predictions = raw_logits[self.dataset.val_mask].argmax(dim=1).cpu().numpy()

        if metric == "accuracy":
            return accuracy_score(all_labels, predictions)
        elif metric == "bacc":
            return balanced_accuracy_score(all_labels, predictions)
        elif metric == "macro_f1":
            return f1_score(all_labels, predictions, average='macro')
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def test(self):
        self.model['default'].eval()
        all_labels = self.dataset.y[self.dataset.test_mask].cpu().numpy()

        with torch.no_grad():
            res = self.model['default'].get_3_embs(self.dataset.x, self.dataset.edge_index)
            raw_logits = res.emb4classi
            predictions = raw_logits[self.dataset.test_mask].argmax(dim=1).cpu().numpy()
            probabilities = torch.nn.functional.softmax(raw_logits[self.dataset.test_mask], dim=1).cpu().numpy()

        accuracy = accuracy_score(all_labels, predictions)
        macro_f1 = f1_score(all_labels, predictions, average='macro')
        bacc = balanced_accuracy_score(all_labels, predictions)
        auc_roc = roc_auc_score(all_labels, probabilities, multi_class='ovr', average='macro')

        return accuracy, bacc, macro_f1, auc_roc
