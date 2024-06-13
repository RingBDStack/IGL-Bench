import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from torch.nn.parameter import Parameter

class graph_convolution_layer(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, device, bias=False):
        super(graph_convolution_layer, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.device = device
        self.weight = Parameter(torch.FloatTensor(in_features, hidden_features))
        self.weight2 = Parameter(torch.FloatTensor(hidden_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters(bias)
        self.mlp_layer_1 = nn.Linear(self.in_features, self.hidden_features, bias=True)
        self.mlp_layer_2 = nn.Linear(self.hidden_features, self.out_features, bias=True)
        self.relu = nn.ReLU()

    def reset_parameters(self, bias):

        nn.init.kaiming_normal_(self.weight)
        nn.init.kaiming_normal_(self.weight2)
        if bias:
            self.bias.data.uniform_(-1, 1)

    def forward(self, adj, features):
        conv_layer_1_output = self.relu(torch.spmm(torch.spmm(adj, features), self.weight))
        conv_layer_2_output = torch.spmm(torch.spmm(adj, conv_layer_1_output), self.weight2)
        self_contribution_layer_output = self.mlp_layer_2(self.relu(self.mlp_layer_1(features)))
        outputs = torch.cat((self_contribution_layer_output, conv_layer_1_output, conv_layer_2_output), dim=1)
        return outputs

class graph_convolution(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim,  device,backbone):
        super(graph_convolution, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device
        if backbone == 'gcn':
            self.gc = graph_convolution_layer(self.input_dim, self.hidden_dim, self.output_dim, self.device)
        elif backbone == 'gin':
            self.gc = GINConv(self.input_dim, self.hidden_dim, self.output_dim)
        else:
            self.gc = SAGE(self.input_dim, self.hidden_dim, self.output_dim)
        self.mlp_1 = nn.Linear(self.hidden_dim + self.output_dim*2, self.hidden_dim)
        # self.mlp_1 = nn.Linear(self.output_dim, self.hidden_dim)
        self.mlp_2 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, features, adj):
        h = self.gc(adj, features)
        h = self.mlp_2(F.relu(self.mlp_1(h)))

        return h


class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
        '''
        super(MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))
    def forward(self, x):
        if self.linear_or_not:
            return self.linear(x)
        else:
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu((self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)
        
class GINConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(GINConv, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GINLayer(input_dim, hidden_dim, hidden_dim))
        for _ in range(1, num_layers-1):
            self.layers.append(GINLayer(hidden_dim, hidden_dim, hidden_dim))
        self.layers.append(GINLayer(hidden_dim, hidden_dim,output_dim))

    def forward(self, adj, x):
        embeddings = []  
        for layer in self.layers:
            x = layer(x, adj)
            embeddings.append(x)  
        return torch.cat(embeddings, dim=1)
        # return x
    
class GINLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GINLayer, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.batchnorm = nn.BatchNorm1d(hidden_dim)
        self.eps = 0
        self.relu = nn.ReLU()

    def forward(self, x, adj):
        neighbor_sum = torch.matmul(adj, x)  
        out = (1 + self.eps) * x + neighbor_sum
        # out = self.batchnorm(self.linear1(out))
        out = self.linear1(out)
        out = self.relu(out) 
        out = self.relu(self.linear2(out)) 
        return out
    
class SAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(SAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.num_layers = num_layers

        # First layer
        self.layers.append(SAGELayer(input_dim, hidden_dim))

        # Hidden layers
        for _ in range(1, num_layers - 1):
            self.layers.append(SAGELayer(hidden_dim, hidden_dim))

        # Output layer
        self.layers.append(SAGELayer(hidden_dim, output_dim))

    def forward(self, adj, x):
        embeddings = []
        for layer in self.layers:
            x = layer(adj, x)
            embeddings.append(x)
        return torch.cat(embeddings,dim=1)

class SAGELayer(nn.Module):
    def __init__(self, in_features, out_features, use_bias=False, activation=nn.ReLU()):
        super(SAGELayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.activation = activation

        # Parameters
        self.weight_neighbor = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight_self = Parameter(torch.FloatTensor(in_features, out_features))
        if self.use_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight_neighbor)
        nn.init.kaiming_normal_(self.weight_self)
        if self.use_bias:
            nn.init.uniform_(self.bias, -1, 1)

    def forward(self, adj, features):
        # Aggregate neighbor features
        neighbor_features = torch.spmm(adj, features)
        # Weighted sum of neighbor features
        neighbors_agg = torch.matmul(neighbor_features, self.weight_neighbor)
        # Weighted sum of self features
        self_features = torch.matmul(features, self.weight_self)

        # Combine using the weights and bias
        output = neighbors_agg + self_features
        if self.use_bias:
            output += self.bias
        return self.activation(output)