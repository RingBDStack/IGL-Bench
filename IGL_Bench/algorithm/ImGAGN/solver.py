import torch
import torch.nn.functional as F
from torch import optim
import numpy as np
import scipy.sparse as sp
from IGL_Bench.algorithm.ImGAGN.utils import euclidean_dist, add_edges, accuracy
from IGL_Bench.algorithm.ImGAGN.models import GCN, Generator

def aug_adj(adj_real_orig, num_generated, features, device):
    num_real_nodes = adj_real_orig.shape[0]
    adj_extended = sp.hstack([adj_real_orig, sp.csr_matrix((num_real_nodes, num_generated))])
    adj_extended = sp.vstack([adj_extended, sp.csr_matrix((num_generated, num_real_nodes + num_generated))])
    adj_extended = adj_extended.tocsr()
    adj_extended.setdiag(1)
    adj = adj_extended
    degree = np.array(adj.sum(axis=1)).flatten()
    deg_inv_sqrt = np.power(degree, -0.5, where=degree > 0)
    deg_inv_sqrt[deg_inv_sqrt == np.inf] = 0.0
    D_inv_sqrt = sp.diags(deg_inv_sqrt)
    adj_norm = D_inv_sqrt.dot(adj).dot(D_inv_sqrt)
    adj_norm = adj_norm.tocoo()
    indices = torch.from_numpy(np.vstack((adj_norm.row, adj_norm.col)).astype(np.int64))
    values = torch.from_numpy(adj_norm.data.astype(np.float32))
    adj_norm = torch.sparse.FloatTensor(indices, values, torch.Size(adj_norm.shape)).to(device)
    return adj_extended, adj_norm

class ImGAGN_node_solver:
    def __init__(self, config, dataset, device='cuda'):
        self.config = config
        self.dataset = dataset
        self.device = device
        self.features = dataset.x
        features = dataset.x
        labels = dataset.y
        num_nodes = features.shape[0]
        self.num_classes = int(labels.max().item() + 1)
        train_idx = torch.nonzero(dataset.train_mask, as_tuple=True)[0]
        train_labels = labels[train_idx]
        class_counts = torch.bincount(train_labels, minlength=self.num_classes)
        nonzero_counts = class_counts.clone()
        self.minority_class = int(torch.argmin(nonzero_counts).item())
        self.minority_count = int(class_counts[self.minority_class].item())
        total_others = int(train_idx.shape[0] - self.minority_count)
        num_other_classes = (self.num_classes - 1) if self.num_classes > 1 else 1
        avg_count = total_others / num_other_classes if num_other_classes > 0 else 0.0
        self.num_generated = int(config.ratio_generated * avg_count)
        if self.num_generated < 0:
            self.num_generated = 0
        self.minority_idx = train_idx[train_labels == self.minority_class]
        self.majority_idx = train_idx[train_labels != self.minority_class]
        self.minority_all_idx = torch.nonzero(labels == self.minority_class, as_tuple=True)[0]
        if self.num_generated > 0:
            self.generate_idx = torch.arange(num_nodes, num_nodes + self.num_generated, dtype=torch.long)
        else:
            self.generate_idx = torch.tensor([], dtype=torch.long)
        self.minority_idx = self.minority_idx.cpu()
        self.majority_idx = self.majority_idx.cpu()
        self.minority_all_idx = self.minority_all_idx.cpu()
        self.generate_idx = self.generate_idx.cpu()
        self.adj_extended, self.adj_norm_extended = aug_adj(dataset.adj, self.num_generated, self.features, self.device)
        self.adj_real = self.adj_extended
        self.adj_norm_orig = self.adj_norm_extended
        self.model = GCN(nfeat=features.shape[1],
                         nhid=config.hidden,
                         nclass=self.num_classes,
                         dropout=config.dropout,
                         generate_node=self.generate_idx,
                         min_node=self.minority_idx)
        self.generator = Generator(self.minority_all_idx.shape[0])
        self.model.to(self.device)
        self.generator.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        if hasattr(dataset, 'adj') and dataset.adj is not None:
            adj_real_orig = dataset.adj
        else:
            import scipy.sparse as sp
            row = dataset.edge_index[0].cpu().numpy()
            col = dataset.edge_index[1].cpu().numpy()
            data = np.ones(row.shape[0])
            num_nodes = features.shape[0]
            adj_real_orig = sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
            adj_sym = adj_real_orig + adj_real_orig.T
            adj_sym.data[adj_sym.data > 1] = 1
            adj_real_orig = adj_sym.tocsr()
        self.features = features.to(self.device)
        self.ori_labels = labels.to(self.device)
        new_labels = torch.full((self.num_generated,), fill_value=self.minority_class, dtype=torch.long).to(self.device)
        self.labels = torch.cat([labels.to(self.device), new_labels], dim=0)
        self.best_state = None
        self.best_val_score = -1.0
        self.best_test_metrics = None

    def reset_parameters(self):
        def reset_module_parameters(module):
            for layer in module.modules():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        reset_module_parameters(self.model)
        reset_module_parameters(self.generator)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        self.best_state = None
        self.best_val_score = -1.0
        self.best_test_metrics = None

    def train(self):
        self.reset_parameters()
        num_folds = 10
        if self.dataset is not None and hasattr(self.dataset, 'name') and self.dataset.name.lower() == 'wiki':
            num_folds = 3
        num_folds = min(num_folds, max(1, len(self.minority_idx)), max(1, len(self.majority_idx)))

        for epoch_gen in range(self.config.epochs_gen):
            fold = epoch_gen % num_folds
            min_start = int(fold * len(self.minority_idx) / num_folds)
            min_end = int((fold + 1) * len(self.minority_idx) / num_folds)
            maj_start = int(fold * len(self.majority_idx) / num_folds)
            maj_end = int((fold + 1) * len(self.majority_idx) / num_folds)
            val_min_idx = self.minority_idx[min_start:min_end]
            val_maj_idx = self.majority_idx[maj_start:maj_end]
            idx_val = torch.cat((val_min_idx, val_maj_idx), dim=0)
            train_min_idx = torch.cat((self.minority_idx[:min_start], self.minority_idx[min_end:]), dim=0)
            if train_min_idx.numel() == 0:
                train_min_idx = self.minority_idx.clone()
            train_maj_idx = torch.cat((self.majority_idx[:maj_start], self.majority_idx[maj_end:]), dim=0)
            if train_maj_idx.numel() == 0:
                train_maj_idx = self.majority_idx.clone()
            idx_train_fold = torch.cat((train_min_idx, train_maj_idx), dim=0)
            if self.num_generated > 0:
                idx_train_fold = torch.cat((idx_train_fold, self.generate_idx), dim=0)

            self.generator.train()
            self.optimizer_G.zero_grad()
            if self.num_generated > 0:
                z = torch.randn((self.num_generated, 100), device=self.device)
            else:
                z = torch.tensor([], device=self.device)
            adj_min_weights = self.generator(z)
            if self.num_generated > 0:
                num_train_min = len(train_min_idx)
                if num_train_min == 0:
                    num_train_min = len(self.minority_idx)
                train_min_mask = torch.zeros(len(self.minority_all_idx), dtype=torch.bool)
                train_min_positions = []
                min_all_positions = {int(node.item()): i for i, node in enumerate(self.minority_all_idx)}
                for node in train_min_idx:
                    if int(node.item()) in min_all_positions:
                        pos = min_all_positions[int(node.item())]
                        train_min_positions.append(pos)
                train_min_positions = sorted(train_min_positions)
                if len(train_min_positions) == 0:
                    train_min_positions = list(range(len(self.minority_all_idx)))
                train_min_positions = torch.tensor(train_min_positions, dtype=torch.long)
                weights_train_min = F.softmax(adj_min_weights[:, train_min_positions], dim=1)
                real_min_feat = self.features[self.minority_all_idx[train_min_positions]].to(self.device)
                gen_features_train = weights_train_min @ real_min_feat
            else:
                gen_features_train = torch.empty((0, self.features.shape[1]), device=self.device)
            weights_all_min = F.softmax(adj_min_weights, dim=1)
            real_all_min_feat = self.features[self.minority_all_idx].to(self.device)
            gen_features_all = weights_all_min @ real_all_min_feat
            gen_weights_matrix = weights_train_min if self.num_generated > 0 else torch.zeros((0,0))
            if self.num_generated > 0:
                matr = gen_weights_matrix.detach().cpu().numpy()
            else:
                matr = np.zeros((0, len(train_min_idx)))
            edges_new_to_min = []
            if matr.size > 0:
                thresh = 1.0 / matr.shape[1] if matr.shape[1] > 0 else 1.0
                new_node_idx_arr, min_node_idx_arr = np.where(matr > thresh)
                if len(new_node_idx_arr) > 0:
                    for new_i, min_j in zip(new_node_idx_arr, min_node_idx_arr):
                        actual_new_node = int(self.generate_idx[new_i].item())
                        actual_min_node = int(train_min_idx[min_j].item()) if min_j < len(train_min_idx) else None
                        if actual_min_node is not None:
                            edges_new_to_min.append((actual_new_node, actual_min_node))
                            edges_new_to_min.append((actual_min_node, actual_new_node))
            if len(edges_new_to_min) > 0:
                edges_new_to_min = np.array(edges_new_to_min).T
                data_new = np.ones(edges_new_to_min.shape[1])
            else:
                edges_new_to_min = np.array([[],[]], dtype=int)
                data_new = np.array([])
            num_total = self.adj_real.shape[0]
            adj_new_edges = sp.coo_matrix((data_new, (edges_new_to_min[0], edges_new_to_min[1])),
                                          shape=(num_total, num_total), dtype=np.float32)
            adj_new_struct = add_edges(self.adj_real, adj_new_edges)
            if not isinstance(adj_new_struct, torch.Tensor):
                adj_new = adj_new_struct.tocsr()
                adj_new.setdiag(1)
                adj_new = adj_new.tocoo()
                deg = np.array(adj_new.sum(axis=1)).flatten()
                deg_inv_sqrt = np.power(deg, -0.5, where=deg>0)
                deg_inv_sqrt[deg_inv_sqrt == np.inf] = 0.0
                D_inv_sqrt = sp.diags(deg_inv_sqrt)
                adj_norm_new = D_inv_sqrt.dot(adj_new).dot(D_inv_sqrt)
                idx_coo = torch.from_numpy(np.vstack((adj_norm_new.row, adj_norm_new.col)).astype(np.int64))
                vals = torch.from_numpy(adj_norm_new.data.astype(np.float32))
                adj_norm_new = torch.sparse.FloatTensor(idx_coo, vals, torch.Size(adj_norm_new.shape))
                adj_norm_new = adj_norm_new.coalesce()
            else:
                adj_norm_new = adj_new_struct.coalesce()
            adj_norm_new = adj_norm_new.to(self.device)
            self.model.eval()
            if self.num_generated > 0:
                features_combined = torch.cat((self.features, gen_features_train.detach()), dim=0)
            else:
                features_combined = self.features
            features_combined = features_combined.to(self.device)
            output_class, output_gen, output_auc = self.model(features_combined, self.adj_norm_orig)
            if self.num_generated > 0:
                labels_fake = torch.zeros(self.num_generated, dtype=torch.long, device=self.device)
                labels_min = torch.full((self.num_generated,), fill_value=self.minority_class, dtype=torch.long, device=self.device)
            else:
                labels_fake = torch.tensor([], dtype=torch.long, device=self.device)
                labels_min = torch.tensor([], dtype=torch.long, device=self.device)
            loss_g_fake = F.nll_loss(output_gen[self.generate_idx.to(self.device)], labels_fake) if self.num_generated > 0 else 0.0
            loss_g_class = F.nll_loss(output_class[self.generate_idx.to(self.device)], labels_min) if self.num_generated > 0 else 0.0
            if self.num_generated > 0:
                loss_g_dist = euclidean_dist(self.features[self.minority_idx].to(self.device), gen_features_train).mean()
            else:
                loss_g_dist = 0.0
            g_loss = loss_g_fake + loss_g_class + loss_g_dist
            g_loss.backward()
            self.optimizer_G.step()

            self.model.train()
            if self.num_generated > 0:
                features_aug = torch.cat((self.features, gen_features_train.detach()), dim=0).to(self.device)
            else:
                features_aug = self.features
            for epoch in range(self.config.epochs):
                output_class, output_gen, output_auc = self.model(features_aug, adj_norm_new)
                class_loss = F.nll_loss(output_class[idx_train_fold.to(self.device)], self.labels[idx_train_fold.to(self.device)])
                if self.num_generated > 0:
                    num_real = idx_train_fold.shape[0] - self.num_generated
                    labels_rf = torch.cat((torch.zeros(num_real, dtype=torch.long), torch.ones(self.num_generated, dtype=torch.long)), dim=0).to(self.device)
                else:
                    labels_rf = torch.zeros(idx_train_fold.shape[0], dtype=torch.long).to(self.device)
                loss_disc = F.nll_loss(output_gen[idx_train_fold.to(self.device)], labels_rf)
                loss_dist = - euclidean_dist(features_aug[self.minority_idx.to(self.device)], features_aug[self.majority_idx.to(self.device)]).mean()
                loss_train = class_loss + loss_disc + loss_dist
                self.optimizer.zero_grad()
                loss_train.backward()
                self.optimizer.step()
                self.model.eval()
                output_class_val, output_gen_val, output_auc_val = self.model(self.features, self.dataset.adj_norm.to(self.device))
                recall_val, f1_val, auc_val, acc_val, _ = accuracy(output_class_val[idx_val.to(self.device)],
                                                                   self.labels[idx_val.to(self.device)],
                                                                   output_auc_val[idx_val.to(self.device)])
                print(f"Epoch {epoch_gen}, Inner Epoch {epoch}: Recall: {recall_val:.4f}, Accuracy: {acc_val:.4f}")
                combined_score = (recall_val + acc_val) / 2.0
                if combined_score > self.best_val_score:
                    self.best_val_score = combined_score
                    self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    output_class_test, output_gen_test, output_auc_test = self.model(self.features, self.dataset.adj_norm.to(self.device))
                    test_recall, test_f1, test_auc, test_acc, _ = accuracy(output_class_test[self.dataset.test_mask].to(self.device),
                                                                           self.ori_labels[self.dataset.test_mask].to(self.device),
                                                                           output_auc_test[self.dataset.test_mask].to(self.device))
                    self.best_test_metrics = {
                        'accuracy': float(test_acc),
                        'balanced_accuracy': float((test_recall + (test_acc if self.num_classes == 2 else 0.0)) / (2 if self.num_classes == 2 else 1)),
                        'macro_f1': float(test_f1),
                        'roc_auc': float(test_auc)
                    }
                self.model.train()
        return None

    def test(self):
        if self.best_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in self.best_state.items()})
        self.model.eval()
        output_class, output_gen, output_auc = self.model(self.features, self.dataset.adj_norm.to(self.device))
        test_idx = torch.nonzero(self.dataset.test_mask, as_tuple=True)[0].to(self.device)
        true_test_labels = self.ori_labels[test_idx]
        pred_test = output_class[test_idx].max(dim=1)[1]
        test_acc = float((pred_test == true_test_labels).sum().item() / len(test_idx))
        from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
        y_true = true_test_labels.cpu().numpy()
        y_pred = pred_test.cpu().numpy()
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        roc_auc = 0.0
        try:
            if self.num_classes > 2:
                y_score = output_class[test_idx].exp().detach().cpu().numpy()
                from sklearn.preprocessing import label_binarize
                Y_true_bin = label_binarize(y_true, classes=list(range(self.num_classes)))
                roc_auc = roc_auc_score(Y_true_bin, y_score, average='macro', multi_class='ovr')
            else:
                y_score = output_class[test_idx].exp()[:, self.minority_class].detach().cpu().numpy()
                roc_auc = roc_auc_score(y_true, y_score)
        except Exception as e:
            roc_auc = float('nan')
        return test_acc, balanced_acc, macro_f1, roc_auc
