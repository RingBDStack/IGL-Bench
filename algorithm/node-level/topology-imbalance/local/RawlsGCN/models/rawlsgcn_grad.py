import torch.nn as nn
import torch.nn.functional as F

from layers.graph_convolution import GraphConvolution


class RawlsGCNGrad(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(RawlsGCNGrad, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        # to fix gradient in trainer
        self.layers_info = {
            "gc1": 0,
            "gc2": 1,
        }

    def forward(self, x, adj):
        pre_act_embs, embs = [], [x]  # adding input node features to make index padding consistent
        x = self.gc1(x, adj)
        x.retain_grad()
        pre_act_embs.append(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        embs.append(x)

        x = self.gc2(x, adj)
        x.retain_grad()
        pre_act_embs.append(x)
        x = F.log_softmax(x, dim=1)
        embs.append(x)
        return pre_act_embs, embs
