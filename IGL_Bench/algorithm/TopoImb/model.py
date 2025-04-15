import torch
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, GlobalAttention
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool
from IGL_Bench.algorithm.TopoImb.layers import MLP, StructConv, MemModule

class StructGraphGNN(torch.nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, n_mem=[8,8,8], dropout=0, nlayer=2, res=True, use_key=False, att='dp'):
        super().__init__()
        self.args = args
        self.nhid = nhid
        self.res = res

        self.pre_conv = nn.Linear(nfeat, nhid)
        self.convs = torch.nn.ModuleList()
        self.mem_cells = torch.nn.ModuleList()
        nlast = nhid
        for layer in range(nlayer):
            mlp = MLP(nlast*2, nhid, nhid, is_cls=False)
            self.convs.append(StructConv(mlp))
            mem = MemModule(nhid, k=n_mem[layer], default=True, use_key=use_key, att=att)
            self.mem_cells.append(mem)

            nlast = nhid

        if res:
            self.lin = Linear(nhid*(nlayer)*2, nclass)
            self.lin_weight = Linear(nhid*(nlayer)*2, 1)
        else:
            self.lin = Linear(nhid*2, nclass)
            self.lin_weight = Linear(nhid*2, 1)

        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        if batch is None: # No batch given
            print('no batch info given')
            batch = x.new(x.size(0)).long().fill_(0)

        x = self.embedding(x,edge_index,edge_weight,batch=batch, return_list=False, graph_level=True)

        x = self.lin(x)

        return F.log_softmax(x, dim=1)

    
    def embedding(self, x, edge_index, edge_weight = None, return_list=False, batch=None, return_logits=False, return_sim=False,graph_level=False):
        assert not (return_sim and return_list), "can not return both sim and list"
        assert not (return_sim and return_logits), "can not return both sim and logits"

        x = self.pre_conv(x)
        pre_x = x
        xs = []
        sim_s = []
        for gconv, mem_cell in zip(self.convs,self.mem_cells):
            x = gconv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)

            x, sim = mem_cell(x, return_s=True)# shape: ()
            '''
            graph_sim = []
            for g_id in set(batch.cpu().numpy()):
                g_sim = sim[batch==g_id].sum(0)
                graph_sim.append(g_sim)
            graph_sim = torch.stack(graph_sim, dim=0)    
            '''
            graph_sim = torch.zeros(len(set(batch.cpu().numpy())),sim.shape[-1], device=x.device).scatter_add_(0, batch.reshape(-1,1).repeat(1, sim.shape[-1]),sim)

            sim_s.append(graph_sim)

            if self.res:
                xs.append(x)

        if self.res:
            x = torch.cat(xs, dim=-1)
            #x = torch.cat([pre_x,x], dim=-1)

        out1 = global_max_pool(x, batch)
        out2 = global_mean_pool(x, batch)        

        if graph_level:
            x = torch.cat([out1,out2],dim=-1)
            emb_list = [x]
        else:
            emb_list = xs

        if return_logits:
            y = self.lin(torch.cat([out1,out2],dim=-1))

            if return_list:
                return emb_list, F.log_softmax(y, dim=1)
            else:
                return x, F.log_softmax(y, dim=1)
        else:
            if return_list:
                return emb_list

        if return_sim:
            return x, sim_s
        else:
            return x

    def predict_weight(self, x, edge_index, edge_weight=None,  batch=None,return_sim=False):
        if return_sim:
            x, sim_s = self.embedding(x,edge_index,edge_weight,batch=batch, return_sim=return_sim, graph_level=True)
        else:
            x = self.embedding(x,edge_index,edge_weight, batch=batch, return_sim=return_sim, graph_level=True)

        x = self.lin_weight(x)

        x = torch.sigmoid(x)
        norm = x.shape[0]/x.sum().detach()
        x = x*norm

        if return_sim:
            return x, sim_s
        else:
            return x

class GraphGCN(torch.nn.Module):
    def __init__(self, args, n_feat, n_hidden, n_class, nlayer=2, dropout=0.5, res=False):
        super().__init__()
        self.args = args
        self.n_hidden = n_hidden
        self.res = res

        self.convs = torch.nn.ModuleList()
        nlast = n_feat
        for layer in range(nlayer):
            self.convs.append(GCNConv(nlast, n_hidden))
            nlast = n_hidden

        if res:
            self.lin = Linear(n_hidden * nlayer * 2, n_class)
        else:
            self.lin = Linear(n_hidden * 2, n_class)

        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        if batch is None:  # No batch given
            print('no batch info given')
            batch = x.new(x.size(0)).long().fill_(0)

        # Ensure x, edge_index, and batch are on the same device
        x = x.to(edge_index.device)
        if edge_weight is not None:
            edge_weight = edge_weight.to(edge_index.device)
        batch = batch.to(edge_index.device)

        x = self.embedding(x, edge_index, edge_weight, batch=batch, graph_level=True)

        x = self.lin(x)
        return F.log_softmax(x, dim=1)

    def embedding(self, x, edge_index, edge_weight=None, batch=None, return_list=False, return_logits=False, graph_level=False):
        xs = []
        for gconv in self.convs:
            x = gconv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            if self.res:
                xs.append(x)

        if self.res:
            x = torch.cat(xs, dim=-1)


        out1 = global_max_pool(x, batch)
        out2 = global_mean_pool(x, batch)

        if graph_level:
            emb_list = [out1, out2]
            x = torch.cat([out1, out2], dim=-1)
        else:
            emb_list = xs

        if return_logits:
            y = self.lin(torch.cat([out1, out2], dim=-1))
            if return_list:
                return emb_list, F.log_softmax(y, dim=1)
            else:
                return x, F.log_softmax(y, dim=1)
        else:
            if return_list:
                return emb_list
            else:
                return x

class GraphGIN(torch.nn.Module):
    def __init__(self, args, n_feat, n_hidden, n_class, dropout, nlayer=2, res=True):
        super().__init__()
        self.args = args
        self.n_hidden = n_hidden
        self.res = res

        self.convs = torch.nn.ModuleList()
        nlast = n_feat
        for layer in range(nlayer):
            mlp = MLP(nlast, n_hidden, n_hidden, is_cls=False)
            self.convs.append(GINConv(mlp))
            nlast = n_hidden

        if res:
            self.lin = Linear(n_hidden*nlayer*2, n_class)
        else:
            self.lin = Linear(n_hidden*2, n_class)

        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        if batch is None: # No batch given
            print('no batch info given')
            batch = x.new(x.size(0)).long().fill_(0)

        x = self.embedding(x,edge_index,edge_weight, batch=batch, graph_level=True)# 

        x = self.lin(x)
        return F.log_softmax(x, dim=1)

    
    def embedding(self, x, edge_index, edge_weight = None, batch=None, return_list=False, return_logits=False, graph_level=False):

        xs = []
        for gconv in self.convs:
            x = gconv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            if self.res:
                xs.append(x)

        if self.res:
            x = torch.cat(xs, dim=-1)

        out1 = global_max_pool(x, batch)
        out2 = global_mean_pool(x, batch)
        
        if graph_level:
            emb_list = [out1, out2]
            x = torch.cat([out1,out2],dim=-1)
        else:
            emb_list = xs

        if return_logits:
            y = self.lin(torch.cat([out1,out2],dim=-1))

            if return_list:
                return emb_list, F.log_softmax(y, dim=1)
            else:
                return x, F.log_softmax(y, dim=1)
        else:
            if return_list:
                return emb_list
            else:
                return x