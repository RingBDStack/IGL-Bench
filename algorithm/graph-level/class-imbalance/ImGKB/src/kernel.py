import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class KerRW(nn.Module):
    def __init__(self, max_step, hidden_graphs, size_hidden_graphs, hidden_dim, device):
        super(KerRW, self).__init__()
        self.max_step = max_step
        self.hidden_graphs = hidden_graphs
        self.size_hidden_graphs = size_hidden_graphs
        self.device = device
        self.adj_hidden = Parameter(
            torch.FloatTensor(hidden_graphs, (size_hidden_graphs * (size_hidden_graphs - 1)) // 2))
        self.features_hidden = Parameter(torch.FloatTensor(hidden_graphs, size_hidden_graphs, hidden_dim))
        self.bn = nn.BatchNorm1d(hidden_graphs * max_step)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.adj_hidden)
        nn.init.kaiming_normal_(self.features_hidden)

    def forward(self, features):

        adj_hidden_norm = torch.zeros(self.hidden_graphs, self.size_hidden_graphs, self.size_hidden_graphs).to(self.device)
        idx = torch.triu_indices(self.size_hidden_graphs, self.size_hidden_graphs, 1)
        adj_hidden_norm[:, idx[0], idx[1]] = self.relu(self.adj_hidden)
        adj_hidden_norm = adj_hidden_norm + torch.transpose(adj_hidden_norm, 1, 2)
        x_o = features
        z = self.features_hidden
        zx = torch.einsum("abc,dc->abd", (z, x_o))
        out = list()
        for i in range(self.max_step):
            z = torch.einsum("abc,acd->abd", (adj_hidden_norm, z))
            t = torch.einsum("abc,dc->abd", (z, x_o))
            t = torch.mul(zx, t)
            t = torch.sum(t, dim=1)
            t = torch.transpose(t, 0, 1) #N X m
            out.append(t)
        out_rw = torch.cat(out, dim=1)
        out_att = torch.stack(out, dim=1)
        return out_rw, out_att