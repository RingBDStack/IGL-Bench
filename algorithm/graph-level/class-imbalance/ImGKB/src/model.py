import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from src.layers import graph_convolution, MLP
from torch_geometric.nn import  global_mean_pool
from src.inforneck import InBo, MI_Est, Attention
from src.kernel import KerRW
import torch.nn.functional as F


class KGIB(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_graphs, size_hidden_graphs, nclass, max_step, num_layers, device,backbone):
        super(KGIB, self).__init__()
        self.num_layers = num_layers
        self.device = device
        self.relu = nn.ReLU()
        self.ker_layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.bn = nn.BatchNorm1d(hidden_graphs * max_step)
        self.infoneck = InBo(hidden_dim + hidden_graphs)
        self.atten = Attention(hidden_graphs)
        self.linear_transform_in = nn.Sequential(nn.Linear(input_dim, 32), nn.Dropout(0.5),self.relu, nn.Linear(32, hidden_dim),self.relu)
        self.linear_transform_out = nn.Sequential(nn.Linear(hidden_graphs * max_step, hidden_graphs * max_step),
                                                  self.relu, nn.Linear(hidden_graphs * max_step, 2))
        self.mlp_1 = nn.Linear(self.num_layers*hidden_graphs * max_step + hidden_dim, hidden_dim)
        self.mlp_2 = nn.Linear(hidden_dim, 2)
        self.SEAG_features = graph_convolution(hidden_dim, hidden_dim, hidden_dim, device,backbone)
        self.conv = nn.ModuleList([self.SEAG_features for _ in range(self.num_layers)])
        for layer in range(self.num_layers):
            self.ker_layers.append(KerRW(max_step, hidden_graphs, size_hidden_graphs, hidden_dim, self.device))
        self.linears_prediction = torch.nn.ModuleList()
        num_mlp_layers = 2
        hidden_dim1 = hidden_graphs * max_step
        for layer in range(self.num_layers + 1):
            if layer == 0:
                self.linears_prediction.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, nclass))
            else:
                self.linears_prediction.append(MLP(num_mlp_layers, hidden_dim1, hidden_dim, nclass))

    def forward(self, adj, features, graph_indicator):
        h = self.linear_transform_in(features)
        graph_embs = global_mean_pool(h, graph_indicator)
        hidden_rep = [graph_embs]
        loss_mi = 0
        for layer in range(self.num_layers):
            h = self.conv[layer](h, adj)
            h_g = global_mean_pool(h,graph_indicator)
            h1, h_att = self.ker_layers[layer](h_g)
            h_a,_ =self.atten(h_att)
            loss_mi += MI_Est(self.infoneck, h_g, h_a)  # I(H_G, H^_G)
            hidden_rep.append(h1)

        # I(Y, H^_G)
        score_over_layer = 0
        for layer, h in enumerate(hidden_rep):
            score_over_layer += self.linears_prediction[layer](h)

        return score_over_layer, loss_mi