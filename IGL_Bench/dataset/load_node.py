import scipy.sparse as sp
import numpy as np
import torch
import random
import os
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Amazon,Actor,WikipediaNetwork
from ogb.nodeproppred import PygNodePropPredDataset
import networkx as nx

def load_node_data(data_name, data_path):
    if data_name == 'ogbn-arxiv':
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..',data_path, 'ogb')
    else:
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..',data_path, 'pyG')
    
    data_path = os.path.join(path, data_name)
    data_dict = {'Cora':'planetoid','CiteSeer':'planetoid','PubMed':'planetoid',
                'Photo':'amazon','Computers':'amazon','Actor':'Actor',
                'Chameleon':'WikipediaNetwork','Squirrel':'WikipediaNetwork','ogbn-arxiv':'ogbn'}    
    target_type = data_dict[data_name]
    if target_type == 'amazon':
        target_dataset = Amazon(data_path, name=data_name)
    elif target_type == 'planetoid':
        target_dataset = Planetoid(data_path, name=data_name)
    elif target_type == 'WikipediaNetwork':
         target_dataset = WikipediaNetwork(root=data_path, name=data_name, geom_gcn_preprocess=True)    
    elif target_type == 'Actor':
        target_dataset = Actor(data_path)
    elif data_name == 'ogbn-arxiv':
        target_dataset = PygNodePropPredDataset(root=data_path, name='ogbn-arxiv')
    
    target_data=target_dataset[0]
    features = target_data.x

    if data_name in ['Cora',"CiteSeer"]:
        features = normalize_features(features)
        features = torch.FloatTensor(np.array(features))
        
    if data_name not in ['ogbn-arxiv','PubMed']:
        adj = index2dense(target_data.edge_index,target_data.num_nodes)
        adj = nx.adjacency_matrix(nx.from_numpy_array(adj))
        adj = adj + sp.eye(adj.shape[0])
        adj_norm = normalize_sparse_adj(adj)
        adj_norm = torch.Tensor(adj_norm.todense())
        adj = torch.Tensor(adj.todense())
        target_data.adj = adj
        target_data.adj_norm = adj_norm
    
    target_data.x = features
    
    if target_data.y.dim() == 2:
        if target_data.y.size(1) > 1:
            target_data.y = target_data.y.argmax(dim=1)
        else:
            target_data.y = target_data.y.squeeze(1)
    
    return target_data

def normalize_sparse_adj(mx):
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

def normalize_features(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def index2dense(edge_index, nnode):
    idx = edge_index.numpy()
    adj = np.zeros((nnode,nnode))
    adj[(idx[0], idx[1])] = 1
    sum = np.sum(adj)

    return adj