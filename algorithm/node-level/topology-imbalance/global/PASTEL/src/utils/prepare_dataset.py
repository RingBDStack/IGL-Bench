import os
import sys
import numpy as np
import pickle as pkl
import scipy.sparse as sp
import networkx as nx
import torch
import random

from .generic_utils import *
from torch_geometric.datasets import Planetoid, Amazon, WebKB, Actor, WikipediaNetwork, WikiCS


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def normalize_features(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def index2dense(edge_index, nnode=2708):
    idx = edge_index.numpy()
    adj = np.zeros((nnode,nnode))
    adj[(idx[0], idx[1])] = 1
    sum = np.sum(adj)

    return adj


def normalize_adj(adj):
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return torch.FloatTensor(adj)


def load_data(data_dir, dataset_str, shuffle_seed):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name ('cora', 'citeseer', 'pubmed')
    :return: All data input files loaded (as well the training/test data).
    """

    if dataset_str == 'photo' or dataset_str == 'computers' or dataset_str == 'chameleon' or dataset_str == 'squirrel' or dataset_str == 'actor':
        if dataset_str == 'photo' or dataset_str == 'computers':
            target_data = Amazon(data_dir, name=dataset_str)[0]
        elif dataset_str == 'chameleon' or dataset_str == 'squirrel':
            target_data = WikipediaNetwork(root=data_dir, name=dataset_str, geom_gcn_preprocess=True)[0]
        elif dataset_str == 'actor':
            target_data = Actor(root=data_dir)[0]

        target_data.num_classes = np.max(target_data.y.numpy()) + 1

        adj = index2dense(target_data.edge_index,target_data.num_nodes)
        adj = nx.adjacency_matrix(nx.from_numpy_array(adj))

        features = torch.Tensor(target_data.x)

        mask_list = [i for i in range(target_data.num_nodes)]
        random.seed(shuffle_seed)
        random.shuffle(mask_list) 
        labels=target_data.y        
        train_mask_list, valid_mask_list, test_mask_list, target_data.train_node = get_split(mask_list, labels.numpy(), labels.numpy().max().item() + 1)
        train_mask_list.sort()
        valid_mask_list.sort()
        test_mask_list.sort()
        idx_train = torch.LongTensor(train_mask_list)
        idx_val = torch.LongTensor(valid_mask_list)
        idx_test = torch.LongTensor(test_mask_list)

        labels = torch.LongTensor(target_data.y)
        labels_train = labels[idx_train]
        labels_test = labels[idx_test]

    else:
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open(os.path.join(data_dir, 'ind.{}.{}'.format(dataset_str, names[i])), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = parse_index_file(os.path.join(data_dir, 'ind.{}.test.index'.format(dataset_str)))
        test_idx_range = np.sort(test_idx_reorder)

        ty_tmp = ty
        if dataset_str == 'citeseer':
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range-min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range-min(test_idx_range), :] = ty
            ty = ty_extended

        raw_features = sp.vstack((allx, tx)).tolil()
        raw_features[test_idx_reorder, :] = raw_features[test_idx_range, :]
        features = normalize_features(raw_features)
        raw_features = torch.Tensor(raw_features.todense())
        features = torch.Tensor(features.todense())

        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

        labels = np.vstack((ally, ty))
        all_idx = [i for i in range(len(labels))]
        labels[test_idx_reorder, :] = labels[test_idx_range, :]
        labels = torch.LongTensor(np.argmax(labels, axis=1))
        random.seed(shuffle_seed)
        random.shuffle(all_idx )   
        idx_train, idx_val, idx_test,_ = get_split(all_idx, labels.numpy(), labels.numpy().max().item() + 1)

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        labels_train = labels[idx_train]
        labels_test = labels[idx_test]

    adj = adj + sp.eye(adj.shape[0])
    adj_norm = normalize_sparse_adj(adj)
    adj_norm = torch.Tensor(adj_norm.todense())

    return adj_norm, features, labels, idx_train, idx_val, idx_test, labels_train, labels_test


def get_split_new(seed, all_idx, labels, n_class):
    labeled_node = [[] for _ in range(n_class)]
    unlabeled_node = [[] for _ in range(n_class)]
    class_nodes = [[] for _ in range(n_class)]
    class_nodes_left = [[] for _ in range(n_class)]

    for i in all_idx:
        label_idx = labels[i]
        class_nodes[label_idx].append(i)

    n_train = 20 * n_class
    n_train_each_class = n_train // n_class
    n_val = 30 * n_class
    n_val_each_class = n_val // n_class

    idx_train = []
    idx_val = []

    for i in range(n_class - 1):
        random.seed(seed + i)
        sampled = random.sample(list(class_nodes[i]), n_train_each_class)
        idx_train.append(np.array(sampled))

    n_train_left = n_train - n_train_each_class * (n_class - 1)
    random.seed(seed + 100)
    sampled = random.sample(list(class_nodes[-1]), n_train_left)
    idx_train.append(np.array(sampled))
    idx_train = np.array(idx_train).flatten()

    left_idx = list(set(list(all_idx)) - set(list(idx_train)))

    for i in left_idx:
        label_idx = labels[i]
        class_nodes_left[label_idx].append(i)

    for i in range(n_class - 1):
        random.seed(seed + i + 50)
        sampled = random.sample(list(class_nodes_left[i]), n_val_each_class)
        idx_val.append(np.array(sampled))

    n_val_left = n_val - n_val_each_class * (n_class - 1)
    random.seed(seed + 627)
    sampled = random.sample(list(class_nodes_left[-1]), n_val_left)
    idx_val.append(np.array(sampled))
    idx_val = np.array(idx_val).flatten()

    idx_test = list(set(list(all_idx)) - set(idx_train) - set(idx_val))

    for iter in idx_train:
        iter_label = labels[iter]
        labeled_node[iter_label].append(iter)

    for iter in left_idx:
        iter_label = labels[iter]
        unlabeled_node[iter_label].append(iter)

    return idx_train, idx_val, idx_test


def get_split(all_idx, all_label, nclass):
    train_each = (len(all_idx)*0.1) // nclass
    valid_each = (len(all_idx)*0.1) // nclass
    train_list = [0 for _ in range(nclass)]
    train_node = [[] for _ in range(nclass)]
    train_idx  = []
    
    for iter1 in all_idx:
        iter_label = all_label[iter1]
        if train_list[iter_label] < train_each:
            train_list[iter_label]+=1
            train_node[iter_label].append(iter1)
            train_idx.append(iter1)

        if sum(train_list)==train_each*nclass:break
    assert sum(train_list)==train_each*nclass
    after_train_idx = list(set(all_idx)-set(train_idx))
    random.shuffle(after_train_idx)
    
    valid_list = [0 for _ in range(nclass)]
    valid_idx  = []
    for iter2 in after_train_idx:
        iter_label = all_label[iter2]
        if valid_list[iter_label] < valid_each:
            valid_list[iter_label]+=1
            valid_idx.append(iter2)
        if sum(valid_list)==valid_each*nclass:break

    assert sum(valid_list)==valid_each*nclass
    test_idx = list(set(after_train_idx)-set(valid_idx))

    return train_idx,valid_idx,test_idx,train_node


def prepare_datasets(config):
    data = {}
    num_nodes = 0

    adj, features, labels, idx_train, idx_val, idx_test, labels_train, labels_test = load_data(config['data_dir'],
                                                                                                                  config['dataset_name'],
                                                                                                                  shuffle_seed=config.get('seed', config['seed']))
    device = config['device']

    data = {'adj': adj.to(device) if device else adj,
            'features': features.to(device) if device else features,
            'labels': labels.to(device) if device else labels,
            'idx_train': idx_train.to(device) if device else idx_train,
            'idx_val': idx_val.to(device) if device else idx_val,
            'idx_test': idx_test.to(device) if device else idx_test,
            'labels_train': labels_train,
            'labels_test': labels_test}

    num_nodes = len(adj)
    return data, num_nodes
