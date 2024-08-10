import csv
from utils import configs as config
import numpy as np
import random
import pickle as pkl
import os
import math
import scipy.sparse as sp
import sys
from scipy.sparse import lil_matrix
import tensorflow as tf

def load_data(dataset, train_ratio, x_flag='feature'):
    """
    Loads input corpus from gcn/data directory

    m.x => the feature vectors of all nodes and labels as scipy.sparse.csr.csr_matrix object;
    m.adj => the adjacency matrix of node-node-label network as scipy.sparse.csr.csr_matrix object;
    m.label => the labels for all nodes as scipy.sparse.csr.csr_matrix objectt;
    train_index.txt => the indices of labeled nodes for supervised training as numpy.ndarray object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    train_ratio =str(train_ratio)
    names = [train_ratio+"_"+x_flag+'.x', train_ratio+'_m.adj', 
             train_ratio+'_m_norm.adj', train_ratio+'_m.label']
    
    objects = []
    for i in range(len(names)):
        with open(os.path.join(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, adj, adj_norm, label = tuple(objects)
    train_indexes = []
    label_counts = {}
    balance_num = 0
    with open(os.path.join(dataset,train_ratio+"_train_index.txt"), 'r') as tir:
        for line in tir:
            params = line.split()
            train_indexes.append(int(params[0]))
            if not params[1] in label_counts.keys():
                label_counts[params[1]] = [int(params[0])]
            else:
                label_counts[params[1]].append(int(params[0]))
    
            if balance_num < len(label_counts[params[1]]):
                balance_num = len(label_counts[params[1]])
    
    # print the class distribution
    label_dist = [(k,len(label_counts[k])) for k in sorted(label_counts.keys())]      
    print('label_distribution:', label_dist)
    print('balance_num:', balance_num)
#         id_text = tir.readline()
#         train_indexes = np.array([int(i) for i in id_text.split()])
    train_indexes = np.array(train_indexes)
    node_num = adj.shape[0]
    temp_indexes = np.setdiff1d(np.array(range(node_num)), train_indexes)
    validation_indexes = np.array(random.sample(list(temp_indexes), int(0.1 * node_num)))
    test_indexes = np.setdiff1d(temp_indexes, validation_indexes)
    
    print(x.shape, adj.shape, label.shape, train_indexes.shape, validation_indexes.shape)
    
    return x, adj, adj_norm, label, train_indexes, validation_indexes, test_indexes

# load_data(config.FILES.cora, str(20)) 

def load_data_boundary(dataset, train_ratio, x_flag='feature'):
    
    train_ratio =str(train_ratio)
    names = [train_ratio+"_"+x_flag+'.x', train_ratio+'_m.adj', 
             train_ratio+'_m_norm.adj', train_ratio+'_m.label']
    
    objects = []
    for i in range(len(names)):
        with open(os.path.join(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, adj, adj_norm, label = tuple(objects)
    train_indexes = []
    label_counts = {}
    balance_num = 0
    with open(os.path.join(dataset,train_ratio+"_train_index.txt"), 'r') as tir:
        for line in tir:
            params = line.split()
            train_indexes.append(int(params[0]))
            if not params[1] in label_counts.keys():
                label_counts[params[1]] = [int(params[0])]
            else:
                label_counts[params[1]].append(int(params[0]))
    
            if balance_num < len(label_counts[params[1]]):
                balance_num = len(label_counts[params[1]])
    
    # print the class distribution
    label_dist = [(k,len(label_counts[k])) for k in sorted(label_counts.keys())]      
    print('label_distribution:', label_dist)
    print('balance_num:', balance_num)
    
    # Sample real nodes for training the GAN model
    real_node_sequence = []
    real_gan_nodes = []
    generated_gan_nodes = []
    # add all labeled nodes for training the gan
    for lab in label_counts.keys():
        nodes = label_counts[lab]
        for no in nodes:
            real_gan_nodes.append([no,lab])
            real_node_sequence.append(no)

        balance_differ = balance_num - len(nodes)
        for i in range(balance_differ):
            idx = random.randint(0, len(nodes)-1)
            real_gan_nodes.append([nodes[idx],lab])
            real_node_sequence.append(nodes[idx])
            
            generated_gan_nodes.append([nodes[idx],lab])

    # shuffle the training samples
    shuffle_indices = np.random.permutation(np.arange(len(real_gan_nodes)))
    real_gan_nodes = [real_gan_nodes[i] for i in shuffle_indices]
    real_node_sequence = [real_node_sequence[i] for i in shuffle_indices]
    print('real_gan_nodes:', real_gan_nodes)
    print('real_node_sequence:', real_node_sequence)
    
    
    train_indexes = np.array(train_indexes)
    node_num = adj.shape[0]
    temp_indexes = np.setdiff1d(np.array(range(node_num)), train_indexes)
    validation_indexes = np.array(random.sample(list(temp_indexes), int(0.1 * node_num)))
    test_indexes = np.setdiff1d(temp_indexes, validation_indexes)
    
    print(x.shape, adj.shape, label.shape, train_indexes.shape, test_indexes.shape, validation_indexes.shape)
    
    # Collect all neighborhood and identically labeled nodes for real nodes
    nodenode_file = os.path.join(dataset, 'adjlist.txt')
    adjlist = {}
    all_neighbor_nodes = []
    with open(nodenode_file, encoding='utf-8') as nnfile:
        for line in nnfile:
            params = line.replace('\n', '').split()
            root = int(params[0])
            neighbors = [int(v) for v in params[1:]]
            adjlist[root] = neighbors    
            
            if root in real_node_sequence:
                for ne in neighbors:
                    if not ne in all_neighbor_nodes:
                        all_neighbor_nodes.append(ne)
    
    real_node_num = len(real_node_sequence)
    real_neighbor_num = len(all_neighbor_nodes)
    adj_neighbor = np.zeros([real_node_num, real_neighbor_num])
    
    for i in range(real_node_num):
        for j in range(real_neighbor_num):
            if all_neighbor_nodes[j] in adjlist[real_node_sequence[i]]:
                adj_neighbor[i][j] = 1
    
    print(adj_neighbor[0:1])
    print(adj_neighbor.shape)
    
    return x, adj, adj_norm, label, train_indexes, test_indexes, validation_indexes, real_gan_nodes, generated_gan_nodes, adj_neighbor, all_neighbor_nodes
            
# load_data_boundary(config.FILES.citeseer, 20) 
