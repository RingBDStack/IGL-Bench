import math

import torch
import torch.nn as nn


class PatternMemory(nn.Module):
    def __init__(self, embeddings_dimension, modelsize = 64):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''

        super(PatternMemory, self).__init__()

        self.know_matrix = nn.Parameter(torch.FloatTensor(modelsize, modelsize))
        self.size = modelsize
        self.dim = embeddings_dimension
        self.leakyrelu = nn.LeakyReLU(0.2)

        self.Wgama = nn.Parameter(torch.FloatTensor(
            1, modelsize))
        self.Wbeta = nn.Parameter(torch.FloatTensor(
            1, modelsize))
        self.Ugama = nn.Parameter(torch.FloatTensor(embeddings_dimension, modelsize))
        self.Ubeta = nn.Parameter(torch.FloatTensor(embeddings_dimension, modelsize))

        self.M = nn.Parameter(torch.FloatTensor(
            modelsize, embeddings_dimension))
        
        self.Eta = nn.Parameter(torch.FloatTensor(modelsize,1))

        self.reset_parameters()

    def reset_parameters(self):
        def reset(tensor):
            stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
            tensor.data.uniform_(-stdv, stdv)
        
        reset(self.know_matrix)
        reset(self.Wgama)
        reset(self.Wbeta)
        reset(self.Ugama)
        reset(self.Ubeta)
        reset(self.M)
        reset(self.Eta)
 
    def forward(self, graph_rep):
        x_g = torch.matmul(graph_rep, self.Ugama)
        x_g = self.leakyrelu(torch.matmul(x_g.unsqueeze(2), self.Wgama).permute(0,2,1))
        x_b = torch.matmul(graph_rep, self.Ubeta)
        x_b = self.leakyrelu(torch.matmul(x_b.unsqueeze(2), self.Wbeta).permute(0,2,1))
        P_q = torch.mul((x_g + 1),self.know_matrix) + x_b
        H_q = self.leakyrelu(torch.matmul(P_q,self.M))
        h_q = H_q.permute(0,2,1).matmul(self.Eta).squeeze(2)

        return h_q
    