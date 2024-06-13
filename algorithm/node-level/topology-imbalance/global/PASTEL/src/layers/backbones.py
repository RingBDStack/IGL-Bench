import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    def __init__(self, in_feats, n_hidden, bias=False, batch_norm=False):
        super(GCNLayer, self).__init__()
        self.weight = torch.Tensor(in_feats, n_hidden)
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


class GCN(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, graph_hops, dropout, batch_norm=False):
        super(GCN, self).__init__()
        self.dropout = dropout

        self.graph_encoders = nn.ModuleList()
        self.graph_encoders.append(GCNLayer(in_feats, n_hidden, batch_norm=batch_norm))

        for _ in range(graph_hops - 2):
            self.graph_encoders.append(GCNLayer(n_hidden, n_hidden, batch_norm=batch_norm))

        self.graph_encoders.append(GCNLayer(n_hidden, n_classes, batch_norm=False))


    def forward(self, x, adj):
        for i, encoder in enumerate(self.graph_encoders[:-1]):
            x = F.relu(encoder(x, adj))
            x = F.dropout(x, self.dropout, training=self.training)

        x = self.graph_encoders[-1](x, adj)
        return x


class GAT(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, num_heads=3, activation=torch.relu, feat_drop=0., attn_drop=0., negative_slope=0.2, residual=False):
        super(GAT, self).__init__()
        self.layers = nn.ModuleList()
        from dgl.nn.pytorch.conv import GATConv

        self.layers.append(GATConv(in_feats, n_hidden, num_heads, feat_drop, attn_drop, negative_slope, residual, activation))
        for _ in range(n_layers - 1):
            self.layers.append(GATConv(n_hidden * num_heads, n_hidden, num_heads, feat_drop, attn_drop, negative_slope, residual, activation))
        self.layers.append(GATConv(n_hidden * num_heads, n_classes, num_heads, feat_drop, attn_drop, negative_slope, residual, None))


    def forward(self, graph, inputs):
        h = inputs
        for l in range(len(self.layers) - 1):
            h = self.layers[l](graph, h).flatten(1)
        logits = self.layers[-1](graph, h).mean(1)

        return logits


class APPNP(nn.Module):
    def __init__(self, in_feats, n_hidden, n_layers, n_classes, activation, feat_drop, edge_drop, alpha, k):
        super(APPNP, self).__init__()
        from dgl.nn.pytorch.conv import APPNPConv
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(in_feats, n_hidden))
        for i in range(n_layers - 1):
            self.layers.append(nn.Linear(n_hidden, n_hidden))
        self.layers.append(nn.Linear(n_hidden, n_classes))
        self.activation = activation
        if feat_drop:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x
        self.propagate = APPNPConv(k, alpha, edge_drop)
        self.reset_parameters()


    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()


    def forward(self, graph, features):
        h = features
        h = self.feat_drop(h)
        h = self.activation(self.layers[0](h))
        for layer in self.layers[1:-1]:
            h = self.activation(layer(h))
        h = self.layers[-1](self.feat_drop(h))
        h = self.propagate(graph, h)

        return h


class GraphSAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout, aggregator_type):
        super(GraphSAGE, self).__init__()
        from dgl.nn.pytorch.conv import SAGEConv

        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type))
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type))
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type))


    def forward(self, graph, features):
        h = self.dropout(features)
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)

        return h


class ChebNet(nn.Module):
    def __init__(self, in_feats, n_classes, n_hidden, n_layers, k, bias):
        super(ChebNet, self).__init__()
        from dgl.nn.pytorch.conv import ChebConv

        self.layers = nn.ModuleList()
        self.layers.append(ChebConv(in_feats, n_hidden, k, bias=bias))
        for _ in range(n_layers - 1):
            self.layers.append(ChebConv(n_hidden, n_hidden, k, bias=bias))
        self.layers.append(ChebConv(n_hidden, n_classes, k, bias=bias))


    def forward(self, graph, features):
        h = features
        for layer in self.layers:
            h = layer(graph, h, [2])

        return h


class SGC(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, k, n_layers, activation, dropout):
        super(SGC, self).__init__()
        from dgl.nn.pytorch.conv import SGConv

        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        self.layers.append(SGConv(in_feats, n_hidden, k, cached=False, bias=False))
        for i in range(n_layers - 1):
            self.layers.append(SGConv(n_hidden, n_hidden, k, cached=False, bias=False))
        self.layers.append(SGConv(n_hidden, n_classes, k, cached=False, bias=False))


    def forward(self, graph, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(graph, h)
            if i != len(self.layers) - 1:
                h = self.activation(h)

        return h