import os
import random

import networkx as nx
import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch_geometric.data import Data, Batch


class SubGraph(object):
    def __init__(self, sample_list = [], unsample_list = []):
        self.sample_list = sample_list
        self.unsample_list = unsample_list

class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''

        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0
        self.edge_mat = 0

        self.max_neighbor = 0
        
        self.nodegroup = 0

        self.K = 0
        self.sample_list = []
        self.unsample_list = []


def load_data(dataset, degree_as_tag):
    '''
        dataset: name of dataset
        test_proportion: ratio of test train split
        seed: random seed for random splitting of dataset
    '''

    print('loading data')
    g_list = []
    label_dict = {}
    feat_dict = {}

    with open('dataset/%s/%s.txt' % (dataset, dataset), 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    # no node attributes
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                if tmp > len(row):
                    node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])
                # print(len(g))
            if node_features != []:
                node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False

            assert len(g) == n

            g_list.append(S2VGraph(g, l, node_tags))

    #add labels and edge_mat       
    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)

        g.label = label_dict[g.label]

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
        g.edge_mat = torch.LongTensor(edges).transpose(0,1)

    if degree_as_tag:
        for g in g_list:
            g.node_tags = list(dict(g.g.degree).values())
            # print(list(dict(g.g.degree).values()))

    #Extracting unique tag labels   
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    print(tagset)

    tag2index = {tagset[i]:i for i in range(len(tagset))}

    for g in g_list:
        g.node_features = torch.zeros(len(g.node_tags), len(tagset))
        g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1


    print('# classes: %d' % len(label_dict))
    print('# maximum node tag: %d' % len(tagset))

    print("# data: %d" % len(g_list))

    return g_list, len(label_dict)


def data_split(graph_list, sample_list, valid_ratio = 0.1, test_ratio = 0.2, seed = 2022):
    random.seed(seed)
    shuffled_indices = list(range(len(graph_list)))
    random.shuffle(shuffled_indices)
    test_set_size = int(len(graph_list)*test_ratio)
    train_set_size = int(len(graph_list)*(1-test_ratio-valid_ratio))
    test_indices = shuffled_indices[-test_set_size:]
    valid_indices = shuffled_indices[train_set_size:-test_set_size]
    train_indices = shuffled_indices[:train_set_size]
    train_graph_list = [graph_list[i] for i in train_indices]
    train_sample_list = [sample_list[i] for i in train_indices]
    test_graph_list = [graph_list[i] for i in test_indices]
    valid_graph_list = [graph_list[i] for i in valid_indices]

    return train_graph_list, valid_graph_list, test_graph_list, train_sample_list


def load_sample(dataset):
    gsample_list = []
    print("Samples loading...")
    rows = 500
    with open('dataset/%s/sampling.txt' % (dataset), 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            K = int(f.readline().strip())
            if K == 0:
                gsample_list.append(SubGraph())
                continue
            sample_list = []
            unsample_list = []
            for j in range(rows):
                row = f.readline().strip().split()
                k = int(row.pop())
                row = [int(n) for n in row]
                sample_list.append(row[:k])
                unsample_list.append(row[k:])
            gsample_list.append(SubGraph(sample_list, unsample_list))
    print("loading finished!")
    return gsample_list


def load_split(load_path='', split_mode='low', load_file=None):
    if load_file:
        if os.path.exists(load_file):
            loaded_split = torch.load(load_file)
        else:
            raise ValueError("Parameter load_file is not a valid file")
    elif os.path.exists(load_path):
        load_file = os.path.join(load_path, 'split_' + split_mode + '.pt')
        if os.path.exists(load_file):
            loaded_split = torch.load(load_file)
        else:
            raise ValueError('Cannot find split_' + split_mode + '.pt in load_path')
    else:
        raise ValueError("Fail to load split file, please check parameter load_path or load_file")
    train_mask = loaded_split['train_mask']
    val_mask = loaded_split['val_mask']
    test_mask = loaded_split['test_mask']
    boundary_size = loaded_split['boundary_size']
    return train_mask, val_mask, test_mask, boundary_size


def preprocess_edge_index(graph):
    edge_index = graph.edge_mat
    num_node = len(graph.node_features)
    self_loop_edge = torch.LongTensor([range(num_node), range(num_node)])
    edge_index = torch.cat([edge_index, self_loop_edge], dim=1)
    return edge_index


def convert_to_pyg_data(batch_graph_list, device):
    pyg_data_list = []
    for graph in batch_graph_list:
        x = graph.node_features.clone().detach().to(torch.float)
        edge_index = preprocess_edge_index(graph)
        data = Data(x=x, edge_index=edge_index)
        pyg_data_list.append(data)

    batched_data = Batch.from_data_list(pyg_data_list)
    return batched_data.to(device)