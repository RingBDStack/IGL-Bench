import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math


class Relation(nn.Module):
    def __init__(self, in_features, ablation):
        super(Relation, self).__init__()

        self.gamma_1 = nn.Linear(in_features, in_features, bias=False)
        self.gamma_2 = nn.Linear(in_features, in_features, bias=False)

        self.beta_1 = nn.Linear(in_features, in_features, bias=False)
        self.beta_2 = nn.Linear(in_features, in_features, bias=False)

        self.r = Parameter(torch.FloatTensor(1, in_features))

        self.elu = nn.ELU()
        self.lrelu = nn.LeakyReLU(0.2)

        self.sigmoid = nn.Sigmoid()
        self.reset_parameter()
        self.ablation = ablation

    def reset_parameter(self):
        stdv = 1. / math.sqrt(self.r.size(1))
        self.r.data.uniform_(-stdv, stdv)

    def forward(self, ft, neighbor):

        if self.ablation == 3:
            self.m = ft + self.r - neighbor
        else:
            gamma = self.gamma_1(ft) + self.gamma_2(neighbor)
            gamma = self.lrelu(gamma) + 1.0

            beta = self.beta_1(ft) + self.beta_2(neighbor)
            beta = self.lrelu(beta)

            self.r_v = gamma * self.r + beta

            # transE
            self.m = ft + self.r_v - neighbor
            '''
            #transH
            norm = F.normalize(self.r_v) 
            h_ft = ft - norm * torch.sum((norm * ft), dim=1, keepdim=True)
            h_neighbor = neighbor - norm * torch.sum((norm * neighbor), dim=1, keepdim=True)
            self.m = h_ft - h_neighbor
            '''
        return self.m  # F.normalize(self.m)


class Relationv2(nn.Module):
    def __init__(self, in_features, out_features, ablation=0):
        super(Relationv2, self).__init__()

        self.gamma1_1 = nn.Linear(in_features, out_features, bias=False)
        self.gamma1_2 = nn.Linear(out_features, in_features, bias=False)

        self.gamma2_1 = nn.Linear(in_features, out_features, bias=False)
        self.gamma2_2 = nn.Linear(out_features, in_features, bias=False)

        self.beta1_1 = nn.Linear(in_features, out_features, bias=False)
        self.beta1_2 = nn.Linear(out_features, in_features, bias=False)

        self.beta2_1 = nn.Linear(in_features, out_features, bias=False)
        self.beta2_2 = nn.Linear(out_features, in_features, bias=False)

        self.r = Parameter(torch.FloatTensor(1, in_features))

        self.ablation = ablation
        self.elu = nn.ELU()
        self.lrelu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()
        self.reset_parameter()

    def weight_init(self, m):
        return

    def reset_parameter(self):
        stdv = 1. / math.sqrt(self.r.size(1))
        self.r.data.uniform_(-stdv, stdv)

    def forward(self, ft, neighbor):

        if self.ablation == 3:
            self.m = ft + self.r - neighbor
        else:

            gamma1 = self.gamma1_2(self.gamma1_1(ft))
            gamma2 = self.gamma2_2(self.gamma2_1(neighbor))
            gamma = self.lrelu(gamma1 + gamma2) + 1.0

            beta1 = self.beta1_2(self.beta1_1(ft))
            beta2 = self.beta2_2(self.beta2_1(neighbor))
            beta = self.lrelu(beta1 + beta2)

            self.r_v = gamma * self.r + beta
            self.m = ft + self.r_v - neighbor

        return F.normalize(self.m)


class Generator(nn.Module):
    def __init__(self, in_features, std, ablation):
        super(Generator, self).__init__()

        self.g = nn.Linear(in_features, in_features, bias=True)
        self.std = std
        self.ablation = ablation

    def forward(self, ft):
        # h_s = ft
        if self.training:
            # if self.ablation == 2:
            mean = torch.zeros(ft.shape, device='cuda')
            ft = torch.normal(mean, 1.)
            # else:
            #    ft = torch.normal(ft, self.std)
        h_s = F.elu(self.g(ft))

        return h_s