import torch
import numpy as np
from scipy.sparse import csr_matrix,lil_matrix
import math

class GraphBatchGenerator:
    def __init__(self, config, adj, features, y, index, device = 'cuda'):

        self.batch_size = config.batch_size
        self.graph_pooling_type = getattr(config,'graph_pooling_type','average')
        self.shuffle = getattr(config,"shuffle", True) 
        self.device = device

        self.adj = [adj[i] for i in index]
        self.features = [features[i] for i in index]
        self.y = [y[i] for i in index]

        self.adj_lst = []
        self.features_lst = []
        self.graph_pool_lst = []
        self.graph_indicator_lst = []
        self.y_lst = []
        self.n_valid_batches = 0

        self.generate_batches()

    def generate_batches(self):
        N = len(self.y)
        if self.shuffle:
            index = np.random.permutation(N)
        else:
            index = np.arange(N, dtype=np.int32)

        n_batches = math.ceil(N / self.batch_size)

        adj_lst_tmp = []
        features_lst_tmp = []
        graph_pool_lst_tmp = []
        graph_indicator_lst_tmp = []
        y_lst_tmp = []

        nu = 0

        for i in range(0, N, self.batch_size):
            n_graphs = min(i + self.batch_size, N) - i
            n_nodes = sum(self.adj[index[j]].shape[0] 
                          for j in range(i, i + n_graphs))

            adj_batch = lil_matrix((n_nodes, n_nodes))
            d_feat = self.features[0].shape[1]
            features_batch = np.zeros((n_nodes, d_feat), dtype=np.float32)

            graph_indicator_batch = np.zeros(n_nodes, dtype=np.int64)
            y_batch = np.zeros(n_graphs, dtype=np.int64)
            graph_pool_batch = lil_matrix((n_graphs, n_nodes))

            idx = 0
            for j in range(i, i + n_graphs):
                n = self.adj[index[j]].shape[0]

                adj_batch[idx: idx + n, idx: idx + n] = self.adj[index[j]]
                features_batch[idx: idx + n, :] = self.features[index[j]]

                graph_indicator_batch[idx: idx + n] = j - i

                y_batch[j - i] = self.y[index[j]]

                if self.graph_pooling_type == "average":
                    graph_pool_batch[j - i, idx: idx + n] = 1.0 / n
                else:
                    graph_pool_batch[j - i, idx: idx + n] = 1

                idx += n

            if sum(y_batch) == 0 or sum(y_batch) == n_graphs:
                nu += 1
            else:
                adj_lst_tmp.append(sparse_mx_to_torch_sparse_tensor(adj_batch).to(self.device))
                features_lst_tmp.append(torch.FloatTensor(features_batch).to(self.device))
                graph_pool_lst_tmp.append(sparse_mx_to_torch_sparse_tensor(graph_pool_batch).to(self.device))
                graph_indicator_lst_tmp.append(torch.LongTensor(graph_indicator_batch).to(self.device))
                y_lst_tmp.append(torch.LongTensor(y_batch).to(self.device))

        self.adj = adj_lst_tmp
        self.features = features_lst_tmp
        self.graph_pool = graph_pool_lst_tmp
        self.graph_indicator = graph_indicator_lst_tmp
        self.y = y_lst_tmp
        self.n_batches = n_batches - nu
        

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row,
                                          sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def my_load_data(dataset):
    edges = dataset.data.edge_index.numpy()  
    graph_indicator = []
    for graph_id, data in enumerate(dataset):
        num_nodes = data.num_nodes
        graph_indicator.extend([graph_id] * num_nodes)
    graph_indicator = np.array(graph_indicator) 

    A = csr_matrix(
        (np.ones(edges.shape[1]), (edges[0, :], edges[1, :])),
        shape=(graph_indicator.size, graph_indicator.size)
    )

    X = dataset.data.x.numpy()  
    labels = [data.y.item() for data in dataset]  # shape: G

    _, graph_size = np.unique(graph_indicator, return_counts=True)
    adj = []
    features = []
    start_idx = 0
    for i in range(len(dataset)):
        end_idx = start_idx + graph_size[i]
        sub_adj = A[start_idx:end_idx, start_idx:end_idx]
        sub_features = X[start_idx:end_idx, :]

        adj.append(sub_adj)
        features.append(sub_features)

        start_idx = end_idx
    labels = np.array(labels)
    return adj, features, labels