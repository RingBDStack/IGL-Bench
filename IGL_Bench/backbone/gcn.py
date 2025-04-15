import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn import global_add_pool

class GCN_graph(torch.nn.Module):
    def __init__(self, n_feat, n_hidden, n_class, n_layer, dropout=0.5, pooling='sum'):
        super(GCN_graph, self).__init__()

        self.n_layer = n_layer
        self.dropout = dropout

        if pooling == 'sum':
            self.pool = global_add_pool
        elif pooling == 'mean':
            self.pool = global_mean_pool
        elif pooling == 'max':
            self.pool = global_max_pool
        else:
            raise ValueError(f"Unsupported pooling method: {pooling}. Choose from 'sum', 'mean', 'max'.")

        self.convs = nn.ModuleList()
        for i in range(n_layer):
            in_dim = n_feat if i == 0 else n_hidden
            self.convs.append(
                GCNConv(in_dim, n_hidden)
            )

        self.out_layer = nn.Linear(n_hidden, n_class)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.pool(x, batch)
        x = self.out_layer(x)
        return x
    
    def encode(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.pool(x, batch)
        return x
    
    def cls(self, encoded_features):
        return self.out_layer(encoded_features)

class GCNLayer(nn.Module):
    def __init__(self, n_feat, n_hidden, bias=False, batch_norm=False):
        super(GCNLayer, self).__init__()
        self.weight = torch.Tensor(n_feat, n_hidden)
        self.weight = nn.Parameter(nn.init.xavier_uniform_(self.weight))
        if bias:
            self.bias = torch.Tensor(n_hidden)
            self.bias = nn.Parameter(nn.init.xavier_uniform_(self.bias))
        else:
            self.register_parameter('bias', None)

        self.bn = nn.BatchNorm1d(n_hidden) if batch_norm else None


    def forward(self, input, adj, batch_norm=True):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            output = output + self.bias
        if self.bn is not None and batch_norm:
            output = self.compute_bn(output)
        return output


    def compute_bn(self, x):
        if len(x.shape) == 2:
            return self.bn(x)
        else:
            return self.bn(x.view(-1, x.size(-1))).view(x.size())


class GCN_node_dense(nn.Module):
    def __init__(self, n_feat, n_hidden, n_class, n_layer, dropout=0.5, batch_norm=False):
        super(GCN_node_dense, self).__init__()
        self.dropout = dropout

        self.graph_encoders = nn.ModuleList()
        self.graph_encoders.append(GCNLayer(n_feat, n_hidden, batch_norm=batch_norm))

        for _ in range(n_layer - 2):
            self.graph_encoders.append(GCNLayer(n_hidden, n_hidden, batch_norm=batch_norm))

        self.graph_encoders.append(GCNLayer(n_hidden, n_class, batch_norm=False))


    def forward(self, x, adj):
        for i, encoder in enumerate(self.graph_encoders[:-1]):
            x = F.relu(encoder(x, adj))
            x = F.dropout(x, self.dropout, training=self.training)

        x = self.graph_encoders[-1](x, adj)
        return x
    
class GCN_node_sparse(nn.Module):
    def __init__(self, n_feat, n_hidden, n_class, n_layer, dropout=0.5, batch_norm=False):
        super(GCN_node_sparse, self).__init__()
        self.dropout = dropout

        self.graph_encoders = nn.ModuleList()
        self.graph_encoders.append(GCNConv(n_feat, n_hidden))

        for _ in range(n_layer - 2):
            self.graph_encoders.append(GCNConv(n_hidden, n_hidden))

        self.graph_encoders.append(GCNConv(n_hidden, n_class))

        self.bn = nn.ModuleList([nn.BatchNorm1d(n_hidden) for _ in range(n_layer - 1)]) if batch_norm else None

    def forward(self, x, edge_index, edge_weight=None):
        for i, encoder in enumerate(self.graph_encoders[:-1]):
            x = encoder(x, edge_index, edge_weight)
            if self.bn is not None:
                x = self.bn[i](x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)

        x = self.graph_encoders[-1](x, edge_index, edge_weight)
        return x