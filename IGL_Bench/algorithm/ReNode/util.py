import torch
import numpy as np

def index2sparse(edge_index, num_nodes):
    # edge_index to sparse format
    row, col = edge_index
    edge_weight = torch.ones(col.size(0), dtype=torch.float32)  # assuming edge weight = 1
    adj_sparse = torch.sparse_coo_tensor(torch.stack([row, col]), edge_weight, (num_nodes, num_nodes))
    return adj_sparse

def direct_sparse_eye(n):
    indices = torch.arange(n)
    indices = torch.stack([indices, indices])
    values = torch.ones(n)
    return torch.sparse_coo_tensor(indices, values, (n, n))

def compute_degree_matrix(A, num_nodes):
    indices = A._indices()
    values = A._values()
    row_indices = indices[0]

    degree = torch.zeros(num_nodes, dtype=values.dtype)

    for idx, value in zip(row_indices, values):
        degree[idx] += value

    degree = degree.pow(-0.5)

    diag_indices = torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)])
    D = torch.sparse_coo_tensor(diag_indices, degree, (num_nodes, num_nodes))

    return D

def index2dense(edge_index,nnode=2708):
    indx = edge_index.numpy()
    adj = np.zeros((nnode,nnode),dtype = 'int8')
    adj[(indx[0],indx[1])]=1
    new_adj = torch.from_numpy(adj).float()
    return new_adj
