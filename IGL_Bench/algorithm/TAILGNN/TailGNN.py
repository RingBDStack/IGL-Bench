import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sp

from IGL_Bench.algorithm.TAILGNN.layers import Relation, Relationv2, Generator
from IGL_Bench.backbone.gcn import GCNLayer


class TailGCN_SP(nn.Module):
    def __init__(self, nfeat, nclass, params, device, ver=1, ablation=0):
        super(TailGCN_SP, self).__init__()

        self.device = device
        self.nhid = params.hidden
        self.dropout = params.dropout
        self.ablation = ablation

        # self.rel1 = TransGCN_SP(nfeat, self.nhid, g_sigma=params.g_sigma, ver=ver)
        if ver == 1:
            self.r1 = Relation(nfeat, ablation=ablation)
        else:
            self.r1 = Relationv2(nfeat, self.nhid, ablation=ablation)
        self.g1 = Generator(nfeat, params.g_sigma, ablation).to(device)

        self.gc1 = GCNLayer(nfeat, self.nhid).to(device)
        self.rel2 = TransGCN_SP(self.nhid, nclass, g_sigma=params.g_sigma, ver=ver, ablation=ablation).to(device)

    def forward(self, x, adj, head, adj_self=None, norm=None):

        # rewrite rel1
        neighbor = sp.mm(adj, x)
        m1 = self.r1(x, neighbor)

        x = x.to(self.device)
        m1 = m1.to(self.device)
        adj = adj.to(self.device)
        adj_self = adj_self.to(self.device)
        norm = norm.to(self.device)

        if head or self.ablation == 2:
            x1 = self.gc1(x, adj_self, norm=norm)
        else:
            if self.ablation == 1:
                h_s = self.g1(m1)
            else:
                h_s = m1

            h_s = torch.mm(h_s, self.gc1.weight)
            h_k = self.gc1(x, adj_self)
            x1 = (h_k + h_s) / (norm + 1)

        x1 = F.elu(x1)
        x1 = F.dropout(x1, self.dropout, training=self.training)

        x2, m2 = self.rel2(x1, adj, adj_self, head, norm)
        norm_m1 = torch.norm(m1, dim=1)
        norm_m2 = torch.norm(m2, dim=1)

        return x2, norm_m1, norm_m2  # , head_prob, tail_prob


class TransGCN_SP(nn.Module):
    def __init__(self, nfeat, nhid, g_sigma, ver, ablation=0):
        super(TransGCN_SP, self).__init__()

        if ver == 1:
            self.r = Relation(nfeat, ablation)
        else:
            self.r = Relationv2(nfeat, nhid, ablation=ablation)

        self.g = Generator(nfeat, g_sigma, ablation)
        self.gc = GCNLayer(nfeat, nhid)
        self.ablation = ablation

    def forward(self, x, adj, adj_self, head, norm):

        # norm = sp.sum(adj, dim=1).to_dense().view(-1,1)
        neighbor = sp.mm(adj, x)
        m = self.r(x, neighbor)

        if head or self.ablation == 2:
            # norm = sp.sum(adj_self, dim=1).to_dense().view(-1,1)
            h_k = self.gc(x, adj_self, norm=norm)
        else:
            if self.ablation == 1:
                h_s = self.g(m)
            else:
                h_s = m

            h_s = torch.mm(h_s, self.gc.weight)
            h_k = self.gc(x, adj_self)
            h_k = (h_k + h_s) / (norm + 1)

        return h_k, m


class Discriminator(nn.Module):
    def __init__(self, in_features):
        super(Discriminator, self).__init__()

        self.d = nn.Linear(in_features, in_features, bias=True)
        self.wd = nn.Linear(in_features, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    '''
    def weight_init(self, m):
        if isinstance(m, Parameter):
            torch.nn.init.xavier_uniform_(m.weight.data)

        if isinstance(m, nn.Linear):
            stdv = 1. / math.sqrt(m.weight.size(1))
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:                
                m.bias.data.uniform_(-stdv, stdv)
    '''

    def forward(self, ft):
        ft = F.elu(ft)
        ft = F.dropout(ft, 0.5, training=self.training)

        fc = F.elu(self.d(ft))
        prob = self.wd(fc)

        return self.sigmoid(prob)

