import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, ReLU, Sequential, ModuleList
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool, global_max_pool

class GIN_graph(torch.nn.Module):
    def __init__(self, n_feat, n_hidden, n_class, n_layer, dropout=0.5, pooling='sum'):
        super(GIN_graph, self).__init__()

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

        self.convs = ModuleList()
        for i in range(n_layer):
            in_dim = n_feat if i == 0 else n_hidden
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(in_dim, n_hidden),
                        BatchNorm1d(n_hidden),
                        ReLU(),
                        Linear(n_hidden, n_hidden),
                        ReLU()
                    )
                )
            )

        self.out_layer = Linear(n_hidden, n_class)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.pool(x, batch)
        x = self.out_layer(x)
        return x
    
    def encode(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.pool(x, batch)
        return x
    
    def cls(self, encoded_features):
        return self.out_layer(encoded_features)