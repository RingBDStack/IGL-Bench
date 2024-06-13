import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import networkx as nx
import multiprocessing as mp
import math
#import dgl
from scipy import sparse
from scipy.sparse import *

from .model import Model
from .utils.generic_utils import to_cuda
from .utils.prepare_dataset import prepare_datasets
from .utils import Timer, DummyLogger, AverageMeter
from .utils import constants as Constants
from .utils.constants import VERY_SMALL_NUMBER, INF


class ModelHandler(object):
    def __init__(self, config):
        self.config_tmp = config
        self.config = config

        # Evaluation Metrics
        self._train_loss = AverageMeter()
        self._dev_loss = AverageMeter()

        self._train_metrics = {'nloss': AverageMeter(),
                               'acc': AverageMeter()}
        self._dev_metrics = {'nloss': AverageMeter(),
                             'acc': AverageMeter()}

        # Logger configuration
        self.logger = DummyLogger(config, dirname=config['out_dir'], pretrained=config['pretrained'])
        self.dirname = self.logger.dirname

        # CUDA configuration
        if not config['no_cuda'] and torch.cuda.is_available():
            self.device = torch.device('cuda' if config['cuda_id'] < 0 else 'cuda:%d' % config['cuda_id'])
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')
        config['device'] = self.device


        #seed = config.get('seed', 42)
        #np.random.seed(3)
        #torch.manual_seed(3)
        #if self.device:
        #    torch.cuda.manual_seed(3)

        # Prepare datasets
        datasets, self.num_nodes = prepare_datasets(config)

        self.num_classes = datasets['labels'].max().item() + 1
        self.n_features_origin = datasets['features'].shape[-1]
        self.raw_features = datasets['features']
        self.dataset_name = config['dataset_name']
        self.whole_loader = datasets
        self.anchor_sets = self.select_anchor_sets(self.whole_loader)

        config['num_anchors'] = self.num_anchors
        config['num_feat'] = datasets['features'].shape[-1]
        config['num_class'] = datasets['labels'].max().item() + 1
        config['num_nodes'] = self.num_nodes

        # Initialize the model
        self.model = Model(config, train_set=datasets.get('train', None))
        self.model.network = self.model.network.to(self.device)

        self._n_test_examples = datasets['idx_test'].shape[0]
        self.run_epoch = self._run_whole_epoch

        self.train_loader = datasets
        self.dev_loader = datasets
        self.test_loader = datasets

        self.config = self.model.config
        self.is_test = False


    def train(self):
        adj = self.whole_loader['adj']
        adj = self.normalize_adj_torch(adj)
        self.cur_adj = adj

        # Calculate the shortest path dists
        self.shortest_path_dists = np.zeros((self.num_nodes, self.num_nodes))
        self.shortest_path_dists = self.cal_shortest_path_distance(self.cur_adj, 5)

        self.shortest_path_dists_anchor = np.zeros((self.num_nodes, self.num_nodes))
        self.shortest_path_dists_anchor = torch.from_numpy(self.cal_spd(self.cur_adj, 0)).to(self.device).to(torch.float32)

        # Calculate group pagerank
        self.group_pagerank_before = self.cal_group_pagerank(adj, self.whole_loader, 0.85)
        self.group_pagerank_after = self.group_pagerank_before
        self.group_pagerank_args = torch.from_numpy(self.cal_group_pagerank_args(self.group_pagerank_before, self.group_pagerank_after)).to(self.device).to(torch.float32)

        # Calculate avg spd before training:
        self.labeled_idx = self.whole_loader['idx_train'].cpu().numpy()
        self.unlabeled_idx = np.append(self.whole_loader['idx_val'].cpu().numpy(), self.whole_loader['idx_test'].cpu().numpy())

        # Check train loader
        if self.train_loader is None or self.dev_loader is None:
            return

        # Set training mode to "train"
        self.is_test = False
        timer = Timer("Train")

        # Initialize results
        self._epoch = self._best_epoch = 0
        self._best_metrics = {}
        for k in self._dev_metrics:
            self._best_metrics[k] = -float('inf')
        self._reset_metrics()

        # Start training until meet the stop condition
        while self._stop_condition(self._epoch, self.config['patience']):
            self._epoch += 1

            # Calculate the shortest path dists every n epochs
            if self._epoch % self.config['pe_every_epochs'] == 0 and self._epoch<self.config['max_epochs']:
                self.position_flag = 1
                self.shortest_path_dists = self.cal_shortest_path_distance(self.cur_adj, 5)
                self.shortest_path_dists_anchor = torch.from_numpy(self.cal_spd(self.cur_adj, 0)).to(self.device).to(torch.float32)
            else:
                self.position_flag = 0

            # Start training this epoch
            self.run_epoch(self.train_loader, training=True, verbose=self.config['verbose'])

            # Log
            if self._epoch % self.config['print_every_epochs'] == 0:
                format_str = "Train Epoch [{} / {}] | Loss: {:0.5f}".format(self._epoch, self.config['max_epochs'], self._train_loss.mean())
                format_str += self.metric_to_str(self._train_metrics)
                self.logger.write_to_file(format_str)
                print(format_str)

            # Validation phase
            dev_output, dev_gold = self.run_epoch(self.dev_loader, training=False, verbose=self.config['verbose'],
                                                  out_predictions=self.config['out_predictions'])

            # dev results
            if self.config['out_predictions']:
                dev_metric_score = self.model.score_func(dev_gold, dev_output)
            else:
                dev_metric_score = None

            # Log
            if self._epoch % self.config['print_every_epochs'] == 0:
                format_str = "  Val Epoch [{} / {}] | Loss: {:0.5f}".format(self._epoch, self.config['max_epochs'], self._dev_loss.mean())
                format_str += self.metric_to_str(self._dev_metrics)
                format_str += '\n'
                self.logger.write_to_file(format_str)
                print(format_str)

            # early stop
            if self.config['early_stop_metric'] == self.model.metric_name and dev_metric_score is not None:
                cur_dev_score = dev_metric_score
            else:
                cur_dev_score = self._dev_metrics[self.config['early_stop_metric']].mean()

            # Evaluate the results and find the best epoch
            if self._best_metrics[self.config['early_stop_metric']] < cur_dev_score:
                self._best_epoch = self._epoch

                for k in self._dev_metrics:
                    self._best_metrics[k] = self._dev_metrics[k].mean()

                if dev_metric_score is not None:
                    self._best_metrics[self.model.metric_name] = dev_metric_score

                if self.config['save_params']:
                    self.model.save(self.dirname)
                    if self._epoch % self.config['print_every_epochs'] == 0:
                        format_str = 'Saved model to {}'.format(self.dirname)
                        self.logger.write_to_file(format_str)
                        print(format_str)

                # Log
                if self._epoch % self.config['print_every_epochs'] == 0:
                    format_str = "Updated: " + self.best_metric_to_str(self._best_metrics)
                    self.logger.write_to_file(format_str)
                    print(format_str)

            self._reset_metrics()

            # Calculate group pagerank
            if self._epoch % self.config['gpr_every_epochs'] == 0 and self._epoch<self.config['max_epochs']:
                self.group_pagerank_after = self.cal_group_pagerank(self.cur_adj, self.whole_loader, 0.85)
                self.group_pagerank_args = torch.from_numpy(self.cal_group_pagerank_args(self.group_pagerank_before, self.group_pagerank_after)).to(self.device).to(torch.float32)

        timer.finish()

        # Log
        format_str = "Finished Training. Training time: {}".format(timer.total) + '\n' + self.summary()
        print(format_str)
        self.logger.write_to_file(format_str)

        return self._best_metrics


    def test(self):
        if self.test_loader is None:
            print("No testing set specified -- skipped testing.")
            return

        # Restore best model
        print('Restoring best model')
        self.model.init_saved_network(self.dirname)
        self.model.network = self.model.network.to(self.device)

        self.is_test = True
        self._reset_metrics()
        timer = Timer("Test")
        for param in self.model.network.parameters():
            param.requires_grad = False

        output, gold = self.run_epoch(self.test_loader, training=False, verbose=0,
                                      out_predictions=self.config['out_predictions'])

        metrics = self._dev_metrics
        format_str = "[test] | "


        test_score = self.model.score_func(gold, output)
        test_wf1_score = self.model.wf1(gold, output)
        test_mf1_score = self.model.mf1(gold, output)
        test_bacc_score = self.model.bacc(gold, output)
        test_auroc_score = self.model.auroc(gold, output)
        format_str = 'seed: {} random_seed: {}\nACC: {:0.5f} | bACC: {:0.5f} | AUROC: {:0.5f} | M-F1: {:0.5f}\n'.format(
        self.config['seed'], self.config['random_seed'],test_score, test_bacc_score, test_auroc_score, test_mf1_score)
        dir = os.path.join('res', self.config["dataset_name"])
        if not os.path.exists(dir):
            os.makedirs(dir)
        filename = os.path.join(dir, f'{self.config["seed"]}.txt')    
        with open(filename, 'a') as file:  
            file.write(format_str)
        print(format_str)
        self.logger.write_to_file(format_str)
        timer.finish()

        format_str = "Finished Testing. Testing time: {}".format(timer.total)
        print(format_str)
        self.logger.write_to_file(format_str)
        self.logger.close()

        test_metrics = {}
        for k in metrics:
            test_metrics[k] = metrics[k].mean()

        if test_score is not None:
            test_metrics[self.model.metric_name] = test_score
        return test_score, test_wf1_score, test_mf1_score


    def one_hot_2_index(self, one_hot_label):
        num_classes = len(one_hot_label)
        for iter in range(num_classes):
            if one_hot_label[iter] == 1:
                return iter


    def cal_group_pagerank(self, adj, data_loader, pagerank_prob):
        num_nodes = self.num_nodes
        num_classes = self.num_classes

        labeled_list = [0 for _ in range(num_classes)]
        labeled_node = [[] for _ in range(num_classes)]
        labeled_node_list = []

        idx_train = data_loader['idx_train']
        labels = data_loader['labels']


        if self.config['dataset_name'] == 'photo' or self.config['dataset_name'] == 'computers' or self.config['dataset_name'] == 'chameleon' or self.config['dataset_name'] == 'squirrel' or self.config['dataset_name'] == 'actor':
            for iter1 in idx_train:
                iter_label_index = labels[iter1]
                labeled_node[iter_label_index].append(iter1)
                labeled_list[iter_label_index] += 1
                labeled_node_list.append(iter1)

        else:
            for iter1 in idx_train:
                iter_label = labels[iter1]
                labeled_node[iter_label].append(iter1)
                labeled_list[iter_label] += 1
                labeled_node_list.append(iter1)

        if (num_nodes > 5000):
            A = adj.detach()
            A_hat = A.to(self.device) + torch.eye(A.size(0)).to(self.device)
            D = torch.sum(A_hat, 1)
            D_inv = torch.eye(num_nodes).to(self.device)

            for iter in range(num_nodes):
                if (D[iter] == 0):
                    D[iter] = VERY_SMALL_NUMBER
                D_inv[iter][iter] = 1.0 / D[iter]
            D = D_inv.sqrt().to(self.device)

            A_hat = torch.mm(torch.mm(D, A_hat), D)
            temp_matrix = torch.eye(A.size(0)).to(self.device) - pagerank_prob * A_hat
            temp_matrix = temp_matrix.cpu().numpy()
            temp_matrix_inv = np.linalg.inv(temp_matrix).astype(np.float32)

            inv = torch.from_numpy(temp_matrix_inv).to(self.device)
            P = (1 - pagerank_prob) * inv

        else:
            A = adj
            A_hat = A.to(self.device) + torch.eye(A.size(0)).to(self.device)
            D = torch.diag(torch.sum(A_hat, 1))
            D = D.inverse().sqrt()
            A_hat = torch.mm(torch.mm(D, A_hat), D)
            P = (1 - pagerank_prob) * ((torch.eye(A.size(0)).to(self.device) - pagerank_prob * A_hat).inverse())

        I_star = torch.zeros(num_nodes)

        for class_index in range(num_classes):
            Lc = labeled_list[class_index]
            Ic = torch.zeros(num_nodes)
            Ic[torch.tensor(labeled_node[class_index])] = 1.0 / Lc
            if class_index == 0:
                I_star = Ic
            if class_index != 0:
                I_star = torch.vstack((I_star,Ic))

        I_star = I_star.transpose(-1, -2).to(self.device)

        Z = torch.mm(P, I_star)
        return Z


    def cal_shortest_path_distance(self, adj, approximate):
        n_nodes = self.num_nodes
        Adj = adj.detach().cpu().numpy()
        G = nx.from_numpy_array(Adj)
        G.edges(data=True)
        dists_array = np.zeros((n_nodes, n_nodes))
        dists_dict = all_pairs_shortest_path_length_parallel(G, cutoff=approximate if approximate > 0 else None)

        cnt_disconnected = 0

        for i, node_i in enumerate(G.nodes()):
            shortest_dist = dists_dict[node_i]
            for j, node_j in enumerate(G.nodes()):
                dist = shortest_dist.get(node_j, -1)
                if dist == -1:
                    cnt_disconnected += 1
                if dist != -1:
                    dists_array[node_i, node_j] = dist
        return dists_array


    def select_anchor_sets(self, data_loader):
        num_classes = self.num_classes
        n_anchors = 0

        class_anchor_num = [0 for _ in range(num_classes)]
        anchor_nodes = [[] for _ in range(num_classes)]
        anchor_node_list = []

        idx_train = data_loader['idx_train']
        labels = data_loader['labels']
        labels_train = data_loader['labels_train']
        range_idx_train = data_loader['idx_train'].shape[0]

        if self.config['dataset_name'] == 'photo' or self.config['dataset_name'] == 'computers' or self.config['dataset_name'] == 'chameleon' or self.config['dataset_name'] == 'squirrel' or self.config['dataset_name'] == 'actor':
            for iter1 in idx_train:
                iter_label_index = labels[iter1]
                anchor_nodes[iter_label_index].append(iter1)
                class_anchor_num[iter_label_index] += 1
                anchor_node_list.append(iter1)
                n_anchors += 1

        else:
            for iter1 in range(range_idx_train):
                iter_label = labels_train[iter1]
                anchor_nodes[iter_label].append(iter1)
                class_anchor_num[iter_label] += 1
                n_anchors += 1
                anchor_node_list.append(iter1)

        self.num_anchors = n_anchors
        self.anchor_node_list = anchor_node_list
        return anchor_nodes


    def cal_node_2_anchor_avg_distance(self, node_index, class_index, anchor_sets, shortest_path_distance_mat):
        spd_sum = 0
        count = len(anchor_sets[class_index])
        for iter in range(count):
            spd_sum += shortest_path_distance_mat[node_index][anchor_sets[class_index][iter]]
        return spd_sum / count


    def cal_shortest_path_distance_anchor(self, adj, anchor_sets, approximate):
        num_nodes = self.num_nodes
        num_classes = self.num_classes
        avg_spd = np.zeros((num_nodes, num_classes))
        shortest_path_distance_mat = self.cal_shortest_path_distance(adj, approximate)
        for iter1 in range(num_nodes):
            for iter2 in range(num_classes):
                avg_spd[iter1][iter2] = self.cal_node_2_anchor_avg_distance(iter1, iter2, anchor_sets, shortest_path_distance_mat)

        max_spd = np.max(avg_spd)
        avg_spd = avg_spd / max_spd

        return avg_spd


    def cal_spd(self, adj, approximate):
        num_anchors = self.num_anchors
        num_nodes = self.num_nodes
        spd_mat = np.zeros((num_nodes, num_anchors))
        shortest_path_distance_mat = self.shortest_path_dists
        for iter1 in range(num_nodes):
            for iter2 in range(num_anchors):
                spd_mat[iter1][iter2] = shortest_path_distance_mat[iter1][self.anchor_node_list[iter2]]

        max_spd = np.max(spd_mat)
        spd_mat = spd_mat / max_spd

        return spd_mat


    def rank_group_pagerank(self, pagerank_before, pagerank_after):
        pagerank_dist = torch.mm(pagerank_before, pagerank_after.transpose(-1, -2)).detach().cpu()
        num_nodes = self.num_nodes
        node_pair_group_pagerank_mat = np.zeros((num_nodes, num_nodes))
        node_pair_group_pagerank_mat_list = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                node_pair_group_pagerank_mat_list.append(pagerank_dist[i, j])
        node_pair_group_pagerank_mat_list = np.array(node_pair_group_pagerank_mat_list)
        index = np.argsort(-node_pair_group_pagerank_mat_list)
        rank = np.argsort(index)
        rank = rank + 1
        iter = 0
        for i in range(num_nodes):
            for j in range(num_nodes):
                node_pair_group_pagerank_mat[i][j] = rank[iter]
                iter = iter + 1

        return node_pair_group_pagerank_mat


    def cal_group_pagerank_args(self, pagerank_before, pagerank_after):
        node_pair_group_pagerank_mat = self.rank_group_pagerank(pagerank_before, pagerank_after) # rank
        num_nodes = self.num_nodes
        PI = 3.1415926
        for i in range(num_nodes):
            for j in range(num_nodes):
                node_pair_group_pagerank_mat[i][j] = 2 - (math.cos((node_pair_group_pagerank_mat[i][j] / (num_nodes * num_nodes)) * PI) + 1)

        return node_pair_group_pagerank_mat


    def rank_group_pagerank_KL(self, pagerank_before, pagerank_after): # KL
        num_nodes = self.num_nodes

        KL_A = pagerank_before[0]
        KL_B = pagerank_after

        for i in range(num_nodes):
            if i == 0:
                for j in range(num_nodes-1):
                    KL_A = torch.vstack((KL_A, pagerank_before[i]))
            else:
                for j in range(num_nodes):
                    KL_A = torch.vstack((KL_A, pagerank_before[i]))

        for i in range(num_nodes-1):
            KL_B = torch.vstack((KL_B, pagerank_after))

        pagerank_dist = F.kl_div(KL_A.softmax(dim=-1).log(), KL_B.softmax(dim=-1), reduction='none').detach()
        pagerank_dist = torch.sum(pagerank_dist, dim=1) * (-1)

        node_pair_group_pagerank_mat_list = pagerank_dist.flatten()
        index = torch.argsort(-node_pair_group_pagerank_mat_list)
        rank = torch.argsort(index)
        rank = rank + 1
        node_pair_group_pagerank_mat = torch.reshape(rank, ((num_nodes, num_nodes)))

        return node_pair_group_pagerank_mat


    def _run_whole_epoch(self, data_loader, training=True, verbose=None, out_predictions=False):
        # Set run mode
        mode = "train" if training else ("test" if self.is_test else "dev")
        self.model.network.train(training)

        # Initialize
        init_adj, features, labels = data_loader['adj'], data_loader['features'], data_loader['labels']
        init_adj = self.normalize_adj_torch(init_adj)

        if mode == 'train':
            idx = data_loader['idx_train']
        elif mode == 'dev':
            idx = data_loader['idx_val']
        else:
            idx = data_loader['idx_test']

        network = self.model.network

        features = F.dropout(features, network.config.get('feat_adj_dropout', 0), training=network.training)
        init_node_vec = features

        # learn graph
        cur_raw_adj, cur_adj = network.learn_graph(network.graph_learner,
                                                   init_node_vec,
                                                   self.shortest_path_dists_anchor,
                                                   self.group_pagerank_args,
                                                   self.position_flag,
                                                   network.graph_skip_conn,
                                                   graph_include_self=network.graph_include_self,
                                                   init_adj=init_adj)

        if self.config['graph_learn'] and self.config.get('max_iter', 10) > 0:
            cur_raw_adj = F.dropout(cur_raw_adj, network.config.get('feat_adj_dropout', 0), training=network.training)
        cur_adj = F.dropout(cur_adj, network.config.get('feat_adj_dropout', 0), training=network.training)

        if network.gnn == 'gcn':
            node_vec = torch.relu(network.encoder.graph_encoders[0](init_node_vec, cur_adj))
            node_vec = F.dropout(node_vec, network.dropout, training=network.training)

            for encoder in network.encoder.graph_encoders[1:-1]:
                node_vec = torch.relu(encoder(node_vec, cur_adj))
                node_vec = F.dropout(node_vec, network.dropout, training=network.training)

            # BP to update weights
            output = network.encoder.graph_encoders[-1](node_vec, cur_adj)
            output = F.log_softmax(output, dim=-1)

        elif network.gnn == 'gat':
            # Convert adj to DGLGraph
            binarized_adj = sparse.coo_matrix(init_adj.detach().cpu().numpy() != 0)
            dgl_graph = dgl.DGLGraph(binarized_adj).to(self.device)

            node_vec = network.encoder.layers[0](dgl_graph, init_node_vec).flatten(1)

            for encoder in network.encoder.layers[1:-1]:
                node_vec_temp = encoder(dgl_graph, node_vec)
                node_vec = node_vec_temp.flatten(1)

            # BP to update weights
            output = network.encoder.layers[-1](dgl_graph, node_vec).mean(1)
            output = F.log_softmax(node_vec, dim=-1)

            node_vec = node_vec_temp.mean(1)

        elif network.gnn == 'appnp':
            binarized_adj = sparse.coo_matrix(init_adj.detach().cpu().numpy() != 0)
            dgl_graph = dgl.DGLGraph(binarized_adj).to(self.device)

            node_vec = network.encoder.feat_drop(init_node_vec)
            node_vec = network.encoder.activation(network.encoder.layers[0](node_vec))

            for encoder in network.encoder.layers[1:-1]:
                node_vec = network.encoder.activation(encoder(node_vec))

            output = network.encoder.layers[-1](network.encoder.feat_drop(node_vec))
            output = network.encoder.propagate(dgl_graph, output)
            output = F.log_softmax(output, dim=-1)

        else:
            # Convert adj to DGLGraph
            binarized_adj = sparse.coo_matrix(init_adj.detach().cpu().numpy() != 0)
            dgl_graph = dgl.DGLGraph(binarized_adj).to(self.device)

            node_vec = network.encoder.layers[0](dgl_graph, init_node_vec)

            for encoder in network.encoder.layers[1:-1]:
                node_vec = encoder(dgl_graph, node_vec)

            # BP to update weights
            output = network.encoder.layers[-1](dgl_graph, node_vec)
            output = F.log_softmax(output, dim=-1)


        # calculate score and loss
        score = self.model.score_func(labels[idx], output[idx])
        loss1 = self.model.criterion(output[idx], labels[idx])

        # graph learn regularization
        if self.config['graph_learn'] and self.config['graph_learn_regularization']:
            loss1 += self.add_graph_loss(cur_raw_adj, init_node_vec)

        # update
        first_raw_adj, first_adj = cur_raw_adj, cur_adj

        # pretrain
        if not mode == 'test':
            if self._epoch > self.config.get('pretrain_epoch', 0):
                max_iter_ = self.config.get('max_iter', 10)
                if self._epoch == self.config.get('pretrain_epoch', 0) + 1:
                    for k in self._dev_metrics:
                        self._best_metrics[k] = -float('inf')
            else:
                max_iter_ = 0
        else:
            max_iter_ = self.config.get('max_iter', 10)

        # set epsilon-NN graph
        if training:
            eps_adj = float(self.config.get('eps_adj', 0))
        else:
            eps_adj = float(self.config.get('test_eps_adj', self.config.get('eps_adj', 0)))

        # update
        pre_raw_adj = cur_raw_adj
        pre_adj = cur_adj

        # reset and iterative
        loss = 0
        iter_ = 0
        while self.config['graph_learn'] and (iter_ == 0 or diff(cur_raw_adj, pre_raw_adj, first_raw_adj).item() > eps_adj) and iter_ < max_iter_:
            torch.cuda.empty_cache()
            iter_ += 1
            pre_adj = cur_adj
            pre_raw_adj = cur_raw_adj
            cur_raw_adj, cur_adj = network.learn_graph(network.graph_learner2,
                                                       node_vec,
                                                       self.shortest_path_dists_anchor,
                                                       self.group_pagerank_args,
                                                       self.position_flag,
                                                       network.graph_skip_conn,
                                                       graph_include_self=network.graph_include_self,
                                                       init_adj=init_adj)

            update_adj_ratio = self.config.get('update_adj_ratio', None)
            update_adj_ratio = math.sin(((self._epoch / self.config['max_epochs']) * 3.1415926)/2) * update_adj_ratio
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

            elif network.gnn == 'gat':
                # Convert adj to DGLGraph
                binarized_adj = sparse.coo_matrix(init_adj.detach().cpu().numpy() != 0)
                dgl_graph = dgl.DGLGraph(binarized_adj).to(self.device)

                node_vec = network.encoder.layers[0](dgl_graph, init_node_vec).flatten(1)

                for encoder in network.encoder.layers[1:-1]:
                    node_vec_temp = encoder(dgl_graph, node_vec)
                    node_vec = node_vec_temp.flatten(1)

                # BP to update weights
                output = network.encoder.layers[-1](dgl_graph, node_vec).mean(1)
                output = F.log_softmax(node_vec, dim=-1)

                node_vec = node_vec_temp.mean(1)

            elif network.gnn == 'appnp':
                binarized_adj = sparse.coo_matrix(init_adj.detach().cpu().numpy() != 0)
                dgl_graph = dgl.DGLGraph(binarized_adj).to(self.device)

                node_vec = network.encoder.feat_drop(init_node_vec)
                node_vec = network.encoder.activation(network.encoder.layers[0](node_vec))

                for encoder in network.encoder.layers[1:-1]:
                    node_vec = network.encoder.activation(encoder(node_vec))

                output = network.encoder.layers[-1](network.encoder.feat_drop(node_vec))
                output = network.encoder.propagate(dgl_graph, output)

                output = F.log_softmax(output, dim=-1)

            else:
                # Convert adj to DGLGraph
                binarized_adj = sparse.coo_matrix(init_adj.detach().cpu().numpy() != 0)
                dgl_graph = dgl.DGLGraph(binarized_adj).to(self.device)

                node_vec = network.encoder.layers[0](dgl_graph, init_node_vec)

                for encoder in network.encoder.layers[1:-1]:
                    node_vec = encoder(dgl_graph, node_vec)

                # BP to update weights
                output = network.encoder.layers[-1](dgl_graph, node_vec)
                output = F.log_softmax(output, dim=-1)


            score = self.model.score_func(labels[idx], output[idx])
            loss += self.model.criterion(output[idx], labels[idx])


            if self.config['graph_learn'] and self.config['graph_learn_regularization']:
                loss += self.add_graph_loss(cur_raw_adj, init_node_vec)
            if self.config['graph_learn'] and not self.config.get('graph_learn_ratio', None) in (None, 0):
                loss += SquaredFrobeniusNorm(cur_raw_adj - pre_raw_adj) * self.config.get('graph_learn_ratio')


        if mode == 'test' and self.config.get('out_raw_learned_adj_path', None):
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
            self.model.clip_grad()  # solve over-fitting
            self.model.optimizer.step()

        self._update_metrics(loss.item(), {'nloss': -loss.item(), self.model.metric_name: score}, 1, training=training)
        self.cur_adj = cur_adj

        return output[idx], labels[idx]


    def metric_to_str(self, metrics):
        format_str = ''
        for k in metrics:
            format_str += ' | {} = {:0.5f}'.format(k.upper(), metrics[k].mean())
        return format_str


    def best_metric_to_str(self, metrics):
        format_str = '\n'
        for k in metrics:
            format_str += ' | {} = {:0.5f}'.format(k.upper(), metrics[k])
        return format_str


    def summary(self):
        start = "\n********************** MODEL SUMMARY **********************"
        info = "Best epoch = {}".format(self._best_epoch) + self.best_metric_to_str(self._best_metrics)
        end = "******************** END MODEL SUMMARY ********************"
        return "\n".join([start, info, end])


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


    def _stop_condition(self, epoch, patience=10):
        """
        Checks have not exceeded max epochs and has not gone patience epochs without improvement.
        """
        no_improvement = epoch >= self._best_epoch + patience
        exceeded_max_epochs = epoch >= self.config['max_epochs']
        return False if exceeded_max_epochs or no_improvement else True


    def add_graph_loss(self, out_adj, features):
        graph_loss = 0
        L = torch.diagflat(torch.sum(out_adj, -1)) - out_adj
        graph_loss += self.config['smoothness_ratio'] * torch.trace(torch.mm(features.transpose(-1, -2), torch.mm(L, features))) / int(np.prod(out_adj.shape))
        ones_vec = to_cuda(torch.ones(out_adj.size(-1)), self.device)
        graph_loss += -self.config['degree_ratio'] * torch.mm(ones_vec.unsqueeze(0), torch.log(torch.mm(out_adj, ones_vec.unsqueeze(-1)) + Constants.VERY_SMALL_NUMBER)).squeeze() / out_adj.shape[-1]
        graph_loss += self.config['sparsity_ratio'] * torch.sum(torch.pow(out_adj, 2)) / int(np.prod(out_adj.shape))
        return graph_loss


    def normalize_adj_torch(self, mx):
        rowsum = mx.sum(1)
        r_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
        r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.
        r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
        mx = torch.matmul(mx, r_mat_inv_sqrt)
        mx = torch.transpose(mx, 0, 1)
        mx = torch.matmul(mx, r_mat_inv_sqrt)
        return mx


def diff(X, Y, Z):
    assert X.shape == Y.shape

    try:
        diff_ = torch.sum(torch.pow(X - Y, 2))
        norm_ = torch.sum(torch.pow(Z, 2))
        diff_ = diff_ / torch.clamp(norm_, min=Constants.VERY_SMALL_NUMBER)
    except:
        X_np = X.cpu().detach().numpy()
        Y_np = Y.cpu().detach().numpy()
        Z_np = Z.cpu().detach().numpy()
        X_Y_np = X_np - Y_np
        X_Y_np_pow = np.power(X_Y_np, 2)
        Z_np_pow = np.power(Z_np, 2)
        diff_np = np.sum(X_Y_np_pow)
        norm_np = np.sum(Z_np_pow)

        diff_ = diff_np / np.clip(a=norm_np, a_min=Constants.VERY_SMALL_NUMBER, a_max=INF)

    return diff_


def batch_diff(X, Y, Z):
    assert X.shape == Y.shape
    diff_ = torch.sum(torch.pow(X - Y, 2), (1, 2)) # Shape: [batch_size]
    norm_ = torch.sum(torch.pow(Z, 2), (1, 2))
    diff_ = diff_ / torch.clamp(norm_, min=Constants.VERY_SMALL_NUMBER)
    return diff_


def SquaredFrobeniusNorm(X):
    return torch.sum(torch.pow(X, 2)) / int(np.prod(X.shape))


def batch_SquaredFrobeniusNorm(X):
    return torch.sum(torch.pow(X, 2), (1, 2)) / int(np.prod(X.shape[1:]))


def single_source_shortest_path_length_range(graph, node_range, cutoff):
    dists_dict = {}
    for node in node_range:
        dists_dict[node] = nx.single_source_shortest_path_length(graph, node, cutoff)   # unweighted
    return dists_dict


def all_pairs_shortest_path_length_parallel(graph, cutoff=None, num_workers=4):
    nodes = list(graph.nodes)
    if len(nodes) < 50:
        num_workers = int(num_workers / 4)
    elif len(nodes) < 400:
        num_workers = int(num_workers / 2)

    pool = mp.Pool(processes=num_workers)
    results = [pool.apply_async(single_source_shortest_path_length_range,
                                args=(graph, nodes[int(len(nodes) / num_workers * i):int(len(nodes) / num_workers * (i + 1))], cutoff)) for i in range(num_workers)]
    output = [p.get() for p in results]
    dists_dict = merge_dicts(output)
    pool.close()
    pool.join()
    return dists_dict


def merge_dicts(dicts):
    result = {}
    for dictionary in dicts:
        result.update(dictionary)
    return result

