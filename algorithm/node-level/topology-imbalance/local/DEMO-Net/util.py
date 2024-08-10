import dgl
import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch
import tensorflow as tf
import time
import pickle
from sklearn import preprocessing

###############################################
# Some code adapted from tkipf/gcn            #
# https://github.com/tkipf/gcn                #
###############################################


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool_)


def load_data(dataset_str, t):
    #TODO dataset
    dataset_name = dataset_str
    masks = torch.load(f'../data_mask/{dataset_name}_{t}_.pth')

    if dataset_name == 'cora':
        dataset = dgl.data.CoraGraphDataset()
        g = dataset[0]
    elif dataset_name == 'citeseer':
        dataset = dgl.data.CiteseerGraphDataset()
        g = dataset[0]
    elif dataset_name == 'pubmed':
        dataset = dgl.data.PubmedGraphDataset()
        g = dataset[0]
    elif dataset_name == 'squirrel' or dataset_name == 'chameleon':
        with open(f'./data/{dataset_name}/out1_node_feature_label.txt', 'r') as file:
            lines = file.readlines()
        feature = []
        labels = []

        for line in lines[1:]:
            parts = line.strip().split('\t')
            node_id = parts[0]
            feature_str = parts[1]
            label = int(parts[2])
            feature_list = list(map(int, feature_str.split(',')))
            feature.append(feature_list)
            labels.append(label)

        features = np.array(feature)
        label_raw = np.array(labels)

        nodes = np.arange(0, len(features))

        with open(f'./data/{dataset_name}/out1_graph_edges.txt', 'r') as file:
            l = file.readlines()
        src, dst = [], []
        for line in l[1:]:
            parts = line.strip().split('\t')
            src.append(int(parts[0]))
            dst.append(int(parts[1]))

    elif dataset_name == 'actor':
        with open(f'./data/{dataset_name}/out1_graph_edges.txt', 'r') as file:
            l = file.readlines()
        src, dst = [], []
        for line in l[1:]:
            parts = line.strip().split('\t')
            src.append(int(parts[0]))
            dst.append(int(parts[1]))

        with open(f'./data/{dataset_name}/out1_node_feature_label.txt', 'r') as file:
            lines = file.readlines()
        feature = []
        labels = []

        for line in lines[1:]:
            parts = line.strip().split('\t')
            node_id = parts[0]
            feature_str = parts[1]
            label = int(parts[2])
            feature_list = np.zeros(931)
            indexs = feature_str.split(',')
            for index in indexs:
                feature_list[int(index)-1] = 1
            feature.append(feature_list)
            labels.append(label)

        features = np.array(feature)
        label_raw = np.array(labels)

        nodes = np.arange(0, len(features))



    elif dataset_name == 'computer':
        dataset = dgl.data.AmazonCoBuyComputerDataset()
        g = dataset[0]
    elif dataset_name == 'photo':
        dataset = dgl.data.AmazonCoBuyPhotoDataset()
        g = dataset[0]
    elif dataset_name == 'arxiv':
        from ogb.nodeproppred import DglNodePropPredDataset
        dataset = DglNodePropPredDataset(name='ogbn-arxiv')
        g = dataset[0][0]
        g.ndata['label'] = dataset[0][1].squeeze()
        from sklearn.preprocessing import StandardScaler
        norm = StandardScaler()
        norm.fit(g.ndata['feat'])
        g.ndata['feat'] = torch.tensor(norm.transform(g.ndata['feat'])).float()
    else:
        return ValueError('Dataset not available')

    if dataset_name == 'cora' or dataset_name == 'citeseer' or dataset_name == 'pubmed' or dataset_name == 'computer' or dataset_name == 'photo' or dataset_name == 'arxiv':
        g = g.remove_self_loop()
        features = g.ndata["feat"].numpy()
        label_raw = g.ndata["label"]
        src, dst = g.edges()
        src = src.numpy()
        dst = dst.numpy()
        nodes = g.nodes().numpy()

    g_edges = list(zip(src, dst))
    lb = preprocessing.LabelBinarizer()
    labels = lb.fit_transform(label_raw)

    G = nx.Graph()
    G.add_edges_from(g_edges)
    adj = nx.adjacency_matrix(G, nodelist=nodes)
    features = sp.csr_matrix(features)

    # Randomly split the train/validation/test set

    train_mask = masks['train_mask']
    val_mask = masks['val_mask']
    test_mask = masks['test_mask']

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    # task information
    degreeNode = np.sum(adj, axis=1).A1
    degreeNode = degreeNode.astype(np.int32)
    degreeValues = set(degreeNode)

    if dataset_name == 'arxiv':
        neighbor_list = []
        degreeTasks = []
        adj = adj.todense()
        for value in degreeValues:
            degreePosition = [int(i) for i, v in enumerate(degreeNode) if v == value]
            degreeTasks.append((value, degreePosition))

        file_path = './arxiv.pickle'
        with open(file_path, 'rb') as f:
                neighbor_list = pickle.load(f)

    else:
        neighbor_list = []
        degreeTasks = []
        adj = adj.todense()
        for value in degreeValues:
            degreePosition = [int(i) for i, v in enumerate(degreeNode) if v == value]
            degreeTasks.append((value, degreePosition))

            d_list = []
            for idx in degreePosition:
                neighs = [int(i) for i in range(adj.shape[0]) if adj[idx, i] > 0]
                d_list += neighs
            neighbor_list.append(d_list)
            assert len(d_list) == value * len(degreePosition), 'The neighbor lists are wrong!'

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, degreeTasks, neighbor_list


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)
    return sparse_mx
