import numpy as np
import scipy as sp
import torch


def link_dropout(adj, idx, k=5):
    tail_adj = adj.copy()
    num_links = np.random.randint(k, size=idx.shape[0])
    num_links += 1

    for i in range(idx.shape[0]):
        index = tail_adj[idx[i]].nonzero()[1]
        new_idx = np.random.choice(index, min(len(index), num_links[i]), replace=False)
        tail_adj[idx[i]] = 0.0
        for j in new_idx:
            tail_adj[idx[i], j] = 1.0
    return tail_adj

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    rowsum = np.where(rowsum==0, 1, rowsum)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)

    return mx

def convert_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

