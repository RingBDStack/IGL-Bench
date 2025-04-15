import torch
from .model import Model
from .cal import cal_shortest_path_distance, cal_group_pagerank_args
from .eval import  AverageMeter
import numpy as np
import torch.nn.functional as F
import math
import os

class PASTEL_node_solver:
    def __init__(self, config, dataset, device='cuda'):
        self.dataset = dataset
        self.config = config
        self.device = device
        self._train_loss = AverageMeter()
        self._dev_loss = AverageMeter()
        self.is_test = False
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.dirname = os.path.join(current_dir, '../../../PASTEL_out', self.dataset.data_name)
        if not os.path.exists(self.dirname):
            os.makedirs(self.dirname)

        self._train_metrics = {'nloss': AverageMeter(),
                               'acc': AverageMeter()}
        self._dev_metrics = {'nloss': AverageMeter(),
                             'acc': AverageMeter()}
        
        self.anchor_sets = self.select_anchor_sets()
        self.initializtion()
        
    def initializtion(self):
        self.model = Model(self.config)
        
    def train(self):
        self.model.reset_parameters()
        self._best_metrics = {}
        for k in self._dev_metrics:
            self._best_metrics[k] = -float('inf')
        self._reset_metrics()
        self.dataset = self.dataset.to('cpu')
        adj = self.dataset.adj_norm
        self.cur_adj = adj
        self.shortest_path_dists = np.zeros((self.config.num_nodes, self.config.num_nodes))
        self.shortest_path_dists = cal_shortest_path_distance(self.cur_adj, 5, self.config.num_nodes)

        self.shortest_path_dists_anchor = np.zeros((self.config.num_nodes, self.config.num_nodes))
        self.shortest_path_dists_anchor = torch.from_numpy(self.cal_spd(self.cur_adj, 0)).to(torch.float32)

        # Calculate group pagerank
        self.group_pagerank_before = self.cal_group_pagerank(adj, 0.85)
        self.group_pagerank_after = self.group_pagerank_before
        self.group_pagerank_args = torch.from_numpy(cal_group_pagerank_args(self.group_pagerank_before, self.group_pagerank_after, self.config.num_nodes)).to(torch.float32)  
        
        self.labeled_idx = self.dataset.train_index
        self.unlabeled_idx = np.append(self.dataset.val_index, self.dataset.test_index)      
        
        num_epochs = getattr(self.config, 'epoch', 500)
        patience = getattr(self.config, 'patience', 10)    

        # Initialize results
        self._epoch = self._best_epoch = 0
        self._best_metrics = {}
        for k in self._dev_metrics:
            self._best_metrics[k] = -float('inf')
        self._reset_metrics()
               
        for epoch in range(1, num_epochs + 1):
            self._epoch = epoch
            if epoch % self.config.pe_every_epochs == 0:
                self.position_flag = 1
                self.cur_adj = self.cur_adj.to('cpu')
                self.shortest_path_dists = cal_shortest_path_distance(self.cur_adj, 5, self.config.num_nodes)
                self.shortest_path_dists_anchor = torch.from_numpy(self.cal_spd(self.cur_adj, 0)).to(torch.float32)
                self.cur_adj = self.cur_adj.to('cuda')
            else:
                self.position_flag = 0    
            
            self.run_epoch(training=True) 
            train_loss = self._train_loss.mean()
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {train_loss:.4f}")
            
            dev_output, dev_gold = self.run_epoch(training=False)    
            dev_metric_score = self.model.score_func(dev_gold, dev_output)
            
            if self.config.early_stop_metric == self.model.metric_name and dev_metric_score is not None:
                cur_dev_score = dev_metric_score
            else:
                cur_dev_score = self._dev_metrics[self.config.early_stop_metric].mean()

            # Evaluate the results and find the best epoch
            if self._best_metrics[self.config.early_stop_metric] < cur_dev_score:
                self._best_epoch = self._epoch

                for k in self._dev_metrics:
                    self._best_metrics[k] = self._dev_metrics[k].mean()

                if dev_metric_score is not None:
                    self._best_metrics[self.model.metric_name] = dev_metric_score

                if self.config.save_params:
                    self.model.save(self.dirname)

            self._reset_metrics()
            
            if epoch > self.config.least_epoch and epoch - self._best_epoch > 10:
                break
            
        print("Training Finished!")               
        
    def cal_spd(self, adj, approximate):
        num_anchors = self.num_anchors
        num_nodes = self.config.num_nodes
        spd_mat = np.zeros((num_nodes, num_anchors))
        shortest_path_distance_mat = self.shortest_path_dists
        for iter1 in range(num_nodes):
            for iter2 in range(num_anchors):
                spd_mat[iter1][iter2] = shortest_path_distance_mat[iter1][self.anchor_node_list[iter2]]

        max_spd = np.max(spd_mat)
        spd_mat = spd_mat / max_spd

        return spd_mat    

    def cal_group_pagerank(self, adj, pagerank_prob):
        num_nodes = self.config.num_nodes
        num_classes = self.config.num_class

        labeled_list = [0 for _ in range(num_classes)]
        labeled_node = [[] for _ in range(num_classes)]
        labeled_node_list = []

        idx_train = torch.LongTensor(self.dataset.train_index)
        labels = self.dataset.y

        for iter1 in idx_train:
            iter_label = labels[iter1]
            labeled_node[iter_label].append(iter1)
            labeled_list[iter_label] += 1
            labeled_node_list.append(iter1)

        if (num_nodes > 5000):
            A = adj.detach()
            A_hat = A + torch.eye(A.size(0))
            D = torch.sum(A_hat, 1)
            D_inv = torch.eye(num_nodes)

            for iter in range(num_nodes):
                if (D[iter] == 0):
                    D[iter] = 1e-12
                D_inv[iter][iter] = 1.0 / D[iter]
            D = D_inv.sqrt()

            A_hat = torch.mm(torch.mm(D, A_hat), D)
            temp_matrix = torch.eye(A.size(0)) - pagerank_prob * A_hat
            temp_matrix = temp_matrix.cpu().numpy()
            temp_matrix_inv = np.linalg.inv(temp_matrix).astype(np.float32)

            inv = torch.from_numpy(temp_matrix_inv)
            P = (1 - pagerank_prob) * inv

        else:
            A = adj
            A_hat = A + torch.eye(A.size(0))
            D = torch.diag(torch.sum(A_hat, 1))
            D = D.inverse().sqrt()
            A_hat = torch.mm(torch.mm(D, A_hat), D)
            P = (1 - pagerank_prob) * ((torch.eye(A.size(0)) - pagerank_prob * A_hat).inverse())

        I_star = torch.zeros(num_nodes)

        for class_index in range(num_classes):
            Lc = labeled_list[class_index]
            Ic = torch.zeros(num_nodes)
            Ic[torch.tensor(labeled_node[class_index])] = 1.0 / Lc
            if class_index == 0:
                I_star = Ic
            if class_index != 0:
                I_star = torch.vstack((I_star,Ic))

        I_star = I_star.transpose(-1, -2)

        Z = torch.mm(P, I_star)
        return Z
    
    def run_epoch(self, training=True):
        mode = "train" if training else ("test" if self.is_test else "dev")
        self.model.network = self.model.network.to(self.device)
        self.dataset = self.dataset.to(self.device)
        self.model.network.train(training)

        # Initialize
        init_adj, features, labels = self.dataset.adj_norm, self.dataset.x, self.dataset.y

        if mode == 'train':
            idx = self.dataset.train_index
        elif mode == 'dev':
            idx = self.dataset.val_index
        else:
            idx = self.dataset.test_index

        network = self.model.network

        features = F.dropout(features, getattr(self.config,'feat_adj_dropout', 0), training=network.training)
        init_node_vec = features

        # learn graph
        cur_raw_adj, cur_adj = network.learn_graph(network.graph_learner,
                                                   init_node_vec,
                                                   self.shortest_path_dists_anchor.to(self.device),
                                                   self.group_pagerank_args.to(self.device),
                                                   self.position_flag,
                                                   network.graph_skip_conn,
                                                   graph_include_self=network.graph_include_self,
                                                   init_adj=init_adj)

        if self.config.graph_learn and getattr(self.config,'max_iter', 10) > 0:
            cur_raw_adj = F.dropout(cur_raw_adj, getattr(self.config,'feat_adj_dropout', 0), training=network.training)
        cur_adj = F.dropout(cur_adj, getattr(self.config,'feat_adj_dropout', 0), training=network.training)

        if network.gnn == 'GCN':
            node_vec = torch.relu(network.encoder.graph_encoders[0](init_node_vec, cur_adj))
            node_vec = F.dropout(node_vec, network.dropout, training=network.training)

            for encoder in network.encoder.graph_encoders[1:-1]:
                node_vec = torch.relu(encoder(node_vec, cur_adj))
                node_vec = F.dropout(node_vec, network.dropout, training=network.training)

            # BP to update weights
            output = network.encoder.graph_encoders[-1](node_vec, cur_adj)
            output = F.log_softmax(output, dim=-1)

        # calculate score and loss
        score = self.model.score_func(labels[idx], output[idx])
        loss1 = self.model.criterion(output[idx], labels[idx])

        # graph learn regularization
        if self.config.graph_learn and self.config.graph_learn_regularization:
            loss1 += self.add_graph_loss(cur_raw_adj, init_node_vec)

        # update
        first_raw_adj, first_adj = cur_raw_adj, cur_adj

        # pretrain
        if not mode == 'test':
            if self._epoch > getattr(self.config,'pretrain_epoch', 0):
                max_iter_ = getattr(self.config,'max_iter', 10)
                if self._epoch == getattr(self.config,'pretrain_epoch', 0) + 1:
                    for k in self._dev_metrics:
                        self._best_metrics[k] = -float('inf')
            else:
                max_iter_ = 0
        else:
            max_iter_ = getattr(self.config,'max_iter', 10)

        # set epsilon-NN graph
        if training:
            eps_adj = float(getattr(self.config,'eps_adj', 0))
        else:
            eps_adj = float(getattr(self.config,'test_eps_adj', getattr(self.config,'eps_adj', 0)))

        # update
        pre_raw_adj = cur_raw_adj
        pre_adj = cur_adj

        # reset and iterative
        loss = 0
        iter_ = 0
        while self.config.graph_learn and (iter_ == 0 or diff(cur_raw_adj, pre_raw_adj, first_raw_adj).item() > eps_adj) and iter_ < max_iter_:
            torch.cuda.empty_cache()
            iter_ += 1
            pre_adj = cur_adj
            pre_raw_adj = cur_raw_adj
            cur_raw_adj, cur_adj = network.learn_graph(network.graph_learner2,
                                                       node_vec,
                                                       self.shortest_path_dists_anchor.to(self.device),
                                                       self.group_pagerank_args.to(self.device),
                                                       self.position_flag,
                                                       network.graph_skip_conn,
                                                       graph_include_self=network.graph_include_self,
                                                       init_adj=init_adj)

            update_adj_ratio = getattr(self.config,'update_adj_ratio', None)
            update_adj_ratio = math.sin(((self._epoch / self.config.epoch) * 3.1415926)/2) * update_adj_ratio
            if update_adj_ratio is not None:
                try:
                    cur_adj = update_adj_ratio * cur_adj + (1 - update_adj_ratio) * first_adj
                except:
                    cur_adj_np = cur_adj.cpu().detach().numpy()
                    first_adj_np = first_adj.cpu().detach().numpy()
                    cur_adj_np = update_adj_ratio * cur_adj_np + (1 - update_adj_ratio) * first_adj_np
                    cur_adj = torch.from_numpy(cur_adj_np).to(self.device)

            if network.gnn == 'gcn':
                node_vec = torch.relu(network.encoder.graph_encoders[0](init_node_vec, cur_adj))
                node_vec = F.dropout(node_vec, network.dropout, training=network.training)

                for encoder in network.encoder.graph_encoders[1:-1]:
                    node_vec = torch.relu(encoder(node_vec, cur_adj))
                    node_vec = F.dropout(node_vec, network.dropout, training=network.training)

                # BP to update weights
                output = network.encoder.graph_encoders[-1](node_vec, cur_adj)
                output = F.log_softmax(output, dim=-1)

            score = self.model.score_func(labels[idx], output[idx])
            loss += self.model.criterion(output[idx], labels[idx])


            if self.config.graph_learn and self.config.graph_learn_regularization:
                loss += self.add_graph_loss(cur_raw_adj, init_node_vec)
            if self.config.graph_learn and not  getattr(self.config,'graph_learn_ratio', None) in (None, 0):
                loss += SquaredFrobeniusNorm(cur_raw_adj - pre_raw_adj) * self.config.get('graph_learn_ratio')


        if mode == 'test' and getattr(self.config,'out_raw_learned_adj_path', None):
            cur_raw_adj = self.normalize_adj_torch(cur_raw_adj)
            out_raw_learned_adj_path = os.path.join(self.dirname, self.config['out_raw_learned_adj_path'])
            np.save(out_raw_learned_adj_path, cur_raw_adj.cpu().detach().numpy())
            print('Saved raw_learned_adj to {}'.format(out_raw_learned_adj_path))

        # calculate loss
        if iter_ > 0:
            loss = loss / iter_ + loss1
        else:
            loss = loss1

        if training:
            self.model.optimizer.zero_grad()
            loss.backward(retain_graph=True) # update weights
            # self.model.clip_grad()  # solve over-fitting
            self.model.optimizer.step()

        self._update_metrics(loss.item(), {'nloss': -loss.item(), self.model.metric_name: score}, 1, training=training)
        self.cur_adj = cur_adj

        return output[idx], labels[idx]

    def test(self):

        # Restore best model
        print('Restoring best model')
        self.model.init_saved_network(self.dirname)
        self.model.network = self.model.network.to(self.device)

        self.is_test = True
        self._reset_metrics()
        for param in self.model.network.parameters():
            param.requires_grad = False

        output, gold = self.run_epoch(training=False)

        test_score = self.model.score_func(gold, output)
        test_wf1_score = self.model.wf1(gold, output)
        test_mf1_score = self.model.mf1(gold, output)
        test_bacc_score = self.model.bacc(gold, output)
        test_auroc_score = self.model.auroc(gold, output)
        
        return test_score, test_bacc_score, test_mf1_score, test_auroc_score

    def add_graph_loss(self, out_adj, features):
        graph_loss = 0
        L = torch.diagflat(torch.sum(out_adj, -1)) - out_adj
        graph_loss += self.config.smoothness_ratio * torch.trace(torch.mm(features.transpose(-1, -2), torch.mm(L, features))) / int(np.prod(out_adj.shape))
        ones_vec = to_cuda(torch.ones(out_adj.size(-1)), self.device)
        graph_loss += -self.config.degree_ratio * torch.mm(ones_vec.unsqueeze(0), torch.log(torch.mm(out_adj, ones_vec.unsqueeze(-1)) + 1e-12)).squeeze() / out_adj.shape[-1]
        graph_loss += self.config.sparsity_ratio * torch.sum(torch.pow(out_adj, 2)) / int(np.prod(out_adj.shape))
        return graph_loss

    def _update_metrics(self, loss, metrics, batch_size, training=True):
        if training:
            if loss:
                self._train_loss.update(loss)
            for k in self._train_metrics:
                if not k in metrics:
                    continue
                self._train_metrics[k].update(metrics[k], batch_size)
        else:
            if loss:
                self._dev_loss.update(loss)
            for k in self._dev_metrics:
                if not k in metrics:
                    continue
                self._dev_metrics[k].update(metrics[k], batch_size)

    def _reset_metrics(self):
        self._train_loss.reset()
        self._dev_loss.reset()

        for k in self._train_metrics:
            self._train_metrics[k].reset()
        for k in self._dev_metrics:
            self._dev_metrics[k].reset()

    def select_anchor_sets(self):
        num_classes = self.dataset.y.numpy().max().item() + 1
        n_anchors = 0

        class_anchor_num = [0 for _ in range(num_classes)]
        anchor_nodes = [[] for _ in range(num_classes)]
        anchor_node_list = []

        idx_train = torch.LongTensor(self.dataset.train_index)
        labels = self.dataset.y
        labels_train = labels[self.dataset.train_index]
        range_idx_train = len(self.dataset.train_index)


        for iter1 in range(range_idx_train):
            iter_label = labels_train[iter1]
            anchor_nodes[iter_label].append(iter1)
            class_anchor_num[iter_label] += 1
            n_anchors += 1
            anchor_node_list.append(iter1)

        self.num_anchors = n_anchors
        self.config.num_anchors = n_anchors
        self.anchor_node_list = anchor_node_list
        self.config.num_class = num_classes
        self.config.num_feat = self.dataset.x.size(1)
        self.config.num_nodes = self.dataset.x.size(0)
        
        return anchor_nodes
    
def to_cuda(x, device=None):
    if device:
        x = x.to(device)
    return x

def diff(X, Y, Z):
    assert X.shape == Y.shape

    try:
        diff_ = torch.sum(torch.pow(X - Y, 2))
        norm_ = torch.sum(torch.pow(Z, 2))
        diff_ = diff_ / torch.clamp(norm_, min=1e-12)
    except:
        X_np = X.cpu().detach().numpy()
        Y_np = Y.cpu().detach().numpy()
        Z_np = Z.cpu().detach().numpy()
        X_Y_np = X_np - Y_np
        X_Y_np_pow = np.power(X_Y_np, 2)
        Z_np_pow = np.power(Z_np, 2)
        diff_np = np.sum(X_Y_np_pow)
        norm_np = np.sum(Z_np_pow)

        diff_ = diff_np / np.clip(a=norm_np, a_min=1e-12, a_max=1e20)

    return diff_


def SquaredFrobeniusNorm(X):
    return torch.sum(torch.pow(X, 2)) / int(np.prod(X.shape))
