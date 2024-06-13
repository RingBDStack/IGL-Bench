import torch
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from torch_geometric.nn import GCNConv, WLConv
import ipdb
from torch_geometric.nn import global_mean_pool, global_max_pool
import torch.nn

class WL_model(torch.nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, dropout, nlayer=2, res=True):
        super().__init__()
        self.args = args
        self.nhid = nhid
        self.res = res

        self.convs = torch.nn.ModuleList()
        for layer in range(nlayer):
            self.convs.append(WLConv())

        self.emb = torch.nn.Embedding(20000,8)

        self.color_size = -1

        self.lin1 = Linear(8, nhid)
        self.lin2 = Linear(nhid, nclass)

        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):

        x = self.embedding(x,edge_index,edge_weight)

        x = self.lin2(F.leaky_relu(self.lin1(x)))

        return F.log_softmax(x, dim=1)

    
    def embedding(self, x, edge_index, edge_weight = None,):

        xs = []
        for gconv in self.convs:
            x = gconv(x, edge_index)
        
        #for node classification, update the color size (topology number)
        if self.color_size == -1:
            self.color_size = len(set(x.cpu().numpy()))

        x = self.emb(x)
        x = F.dropout(x, self.dropout, training=self.training)
        return x

    def wl(self,x, edge_index):
        for gconv in self.convs:
            x = gconv(x, edge_index)

        return x

class WLGraph_model(torch.nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, dropout, nlayer=2, res=True):
        super().__init__()
        self.args = args
        self.nhid = nhid
        self.res = res

        self.convs = torch.nn.ModuleList()
        for layer in range(nlayer):
            self.convs.append(WLConv())

        self.emb = torch.nn.Embedding(5000,32)

        self.color_size = -1
        self.graph_map = {}

        self.lin1 = Linear(32, nhid)
        self.lin2 = Linear(nhid, nclass)

        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        if batch is None: # No batch given
            print('no batch info given')
            ipdb.set_trace()
            batch = x.new(x.size(0)).long().fill_(0)

        x = self.embedding(x,edge_index,edge_weight, batch=batch)

        x = self.lin2(F.leaky_relu(self.lin1(x)))

        return F.log_softmax(x, dim=1)

    def embedding(self, x, edge_index, edge_weight = None,batch=None):
        if x.shape[-1] != 1:
            x = x.argmax(-1)

        xs = []
        for gconv in self.convs:
            x = gconv(x, edge_index)
        
        #for graph classification, update the color size (topology number)
        out = []
        for b_i in set(batch.cpu().numpy()):
            b_idx = batch==b_i
            g_i = x[b_idx]
            idx = hash(tuple(g_i.cpu().numpy().tolist()))
            if idx not in self.graph_map:
                self.graph_map[idx] = len(self.graph_map)
            out.append(self.graph_map[idx])
        g_x = torch.tensor(out, device=x.device)

        if self.color_size == -1:
            self.color_size = len(self.graph_map)

        g_x = self.emb(g_x)
        gx = F.dropout(g_x, self.dropout, training=self.training)
        return gx

    def wl(self,x, edge_index, batch=None):
        if x.shape[-1] != 1:
            x = x.argmax(-1)

        for gconv in self.convs:
            x = gconv(x, edge_index)
        
        out = []
        for b_i in set(batch.cpu().numpy()):
            b_idx = batch==b_i
            g_i = x[b_idx]
            idx = hash(tuple(g_i.cpu().numpy().tolist()))
            if idx not in self.graph_map:
                self.graph_map[idx] = len(self.graph_map)
            out.append(self.graph_map[idx])
        g_x = torch.tensor(out, device=x.device)

        return g_x

    def graph_wl_dist(self,x, edge_index, batch=None):
        # summarize node-wise wl label distribution for each graph
        if x.shape[-1] != 1:
            x = x.argmax(-1)

        for gconv in self.convs:
            x = gconv(x, edge_index)

        out = self.convs[-1].histogram(x, batch, norm=True) # (batch_size, num_colors)
        out = out.to(x.device)

        return out