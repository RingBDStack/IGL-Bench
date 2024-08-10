import dgl
import torch
import math
import torch.nn as nn
import torch.nn.functional as F 

class GCN(torch.nn.Module):
    def __init__(self,
                 in_feats,
                 hidden_lst,
                 dropout=0.5,
                 use_linear=False,
                 norm='identity',
                 prelu=False,
                 encoder_mode=False,
                 mp_norm='both'):
        super(GCN, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.activations = torch.nn.ModuleList()
        self.linears = torch.nn.ModuleList()
        self.in_feats = in_feats
        self.encoder_mode = encoder_mode
        self.hidden_lst = [in_feats] + hidden_lst
        self.use_linear = use_linear

        if norm == 'layer':
            norm = torch.nn.LayerNorm
        elif norm == 'batch':
            norm = torch.nn.BatchNorm1d
        else: 
            norm = torch.nn.Identity

        for in_, out_ in zip(self.hidden_lst[:-1], self.hidden_lst[1:]):
            self.layers.append(dgl.nn.GraphConv(in_, out_, allow_zero_in_degree=True, norm=mp_norm))
            self.norms.append(norm(out_))
            self.activations.append(torch.nn.PReLU() if prelu else torch.nn.ReLU())
            if self.use_linear:
                self.linears.append(torch.nn.Linear(in_, out_, bias=False))

        self.dropout = torch.nn.Dropout(p=dropout)
        self.n_classes = self.hidden_lst[-1]

    def forward(self, g, features, return_all=False):
        h = features
        stack = []
        for i, layer in enumerate(self.layers):
            # dropout
            if i != 0: h = self.dropout(h)

            # apply lnear
            if self.use_linear:
                linear = self.linears[i](h)
            # graph conv
            h = layer(g, h)

            # res
            if self.use_linear:
                h = h + linear

            # activation and norm
            if i != len(self.layers) - 1 or self.encoder_mode:
                h = self.activations[i](self.norms[i](h))
            stack.append(h)
            
        return stack if return_all else stack[-1]
    

class SGC(torch.nn.Module):
    def __init__(self,
                 in_feats,
                 hidden_lst,
                 dropout=0.5,
                 use_linear=False,
                 norm='identity',
                 prelu=False,
                 encoder_mode=False,
                 mp_norm='both'):
        super(SGC, self).__init__()
        self.in_feats = in_feats
        self.encoder_mode = encoder_mode
        self.hidden_lst = [in_feats] + hidden_lst
        self.conv = dgl.nn.SGConv(in_feats, hidden_lst[-1], k=len(self.hidden_lst))

        self.n_classes = self.hidden_lst[-1]

    def forward(self, g, features):
        return self.conv(g, features)
    
class SAGE(torch.nn.Module):
    def __init__(self,
                 in_feats,
                 hidden_lst,
                 dropout=0.5,
                 use_linear=False,
                 norm='identity',
                 prelu=False,
                 encoder_mode=False,
                 mp_norm='both'):
        super(SAGE, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.activations = torch.nn.ModuleList()
        self.linears = torch.nn.ModuleList()
        self.in_feats = in_feats
        self.encoder_mode = encoder_mode
        self.hidden_lst = [in_feats] + hidden_lst
        self.use_linear = use_linear

        if norm == 'layer':
            norm = torch.nn.LayerNorm
        elif norm == 'batch':
            norm = torch.nn.BatchNorm1d
        else: 
            norm = torch.nn.Identity

        for in_, out_ in zip(self.hidden_lst[:-1], self.hidden_lst[1:]):
            self.layers.append(dgl.nn.SAGEConv(in_, out_, aggregator_type='pool'))
            self.norms.append(norm(out_))
            self.activations.append(torch.nn.PReLU() if prelu else torch.nn.ReLU())
            if self.use_linear:
                self.linears.append(torch.nn.Linear(in_, out_, bias=False))

        self.dropout = torch.nn.Dropout(p=dropout)
        self.n_classes = self.hidden_lst[-1]

    def forward(self, g, features, return_all=False):
        h = features
        stack = []
        for i, layer in enumerate(self.layers):
            # dropout
            if i != 0: h = self.dropout(h)

            # apply lnear
            if self.use_linear:
                linear = self.linears[i](h)
            # graph conv
            h = layer(g, h)

            # res
            if self.use_linear:
                h = h + linear

            # activation and norm
            if i != len(self.layers) - 1 or self.encoder_mode:
                h = self.activations[i](self.norms[i](h))
            stack.append(h)
            
        return stack if return_all else stack[-1]

class GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, heads=[8,1], norm=False):
        super().__init__()
        self.gat_layers = nn.ModuleList()
        # two-layer GAT
        self.hidden_lst = [in_size, hid_size]
        self.gat_layers.append(
            dgl.nn.GATConv(
                in_size,
                hid_size,
                heads[0],
                feat_drop=0.6,
                attn_drop=0.6,
                activation=None,
                allow_zero_in_degree=True
            )
        )
        self.gat_layers.append(
            dgl.nn.GATConv(
                hid_size * heads[0],
                out_size,
                heads[1],
                feat_drop=0.6,
                attn_drop=0.6,
                activation=None,
                allow_zero_in_degree=True
            )
        )

        if norm: 
            self.norm = torch.nn.BatchNorm1d(hid_size * heads[0]) 
        else: 
            self.norm = torch.nn.Identity(hid_size * heads[0]) 

    def forward(self, g, inputs):
        h = inputs
        for i, layer in enumerate(self.gat_layers):
            h = layer(g, h)
            if i == 1:  # last layer
                h = h.mean(1)
            else:  # other layer(s)
                h = F.elu(self.norm(h.flatten(1)))
        return h

class Generator(torch.nn.Module):
    def __init__(self,
                 dropout,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 args,
                 three_layer=False,
                 norm='identity',
                 fusion='cat',
                 use_linear=False,
                 mp_norm='right',
                 gnn='gcn'
                 ):
        super(Generator, self).__init__()
        if gnn =='gcn':
            self.generator = GCN(input_dim, [hidden_dim, hidden_dim if fusion == 'cat' else input_dim], dropout, \
                                prelu=False, norm=norm, encoder_mode=False, use_linear=use_linear, mp_norm=mp_norm)
        elif gnn =='sage':
            self.generator = SAGE(input_dim, [hidden_dim, hidden_dim if fusion == 'cat' else input_dim], dropout, \
                                prelu=False, norm=norm, encoder_mode=False, use_linear=use_linear, mp_norm=mp_norm)
        elif gnn =='gat':
            self.generator = GAT(input_dim, hidden_dim, hidden_dim)
        
        if norm == 'layer':
            self.last_norm = torch.nn.LayerNorm(hidden_dim if fusion == 'cat' else input_dim)  
        elif norm == 'batch':
            self.last_norm = torch.nn.BatchNorm1d(hidden_dim if fusion == 'cat' else input_dim)  
        else:
            self.last_norm = torch.nn.Identity()
        self.last_norm = torch.nn.LayerNorm(hidden_dim if fusion == 'cat' else input_dim) if args.dataset =='arxiv' else torch.nn.Identity()
        self.fusion = fusion

        if fusion == 'cat':
            self.generator_MLP = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim+input_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_dim, output_dim, bias=False)
            )
        else:
            self.generator_MLP = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_dim, output_dim)
                )
        
        for m in self.modules():
            self.weights_init(m)


    def weights_init(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, g, masked_offset):
        h = self.generator(g, g.ndata['feat'])
        # h = self.last_norm(h)
        if self.fusion == 'sum':
            h = torch.nn.functional.relu(self.last_norm(h[masked_offset]+g.ndata['feat'][masked_offset]))
            return self.generator_MLP(h)
        elif self.fusion == 'product': 
            h = torch.nn.functional.relu(self.last_norm(h[masked_offset]*g.ndata['feat'][masked_offset]))
            return self.generator_MLP(h)
        else:
            h = torch.nn.functional.relu(self.last_norm(h[masked_offset]))
            return self.generator_MLP(
                torch.cat([h, g.ndata['feat'][masked_offset]], dim=1)
                ) 