import torch
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, GlobalAttention
from models.layers import StructConv, MemModule
import ipdb
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool
from models.layers import MLP


def manual_global_max_pool(x, batch):
    batch_size = batch.max().item() + 1
    pooled = torch.zeros((batch_size, x.size(1)), device=x.device)
    for i in range(batch_size):
        pooled[i] = x[batch == i].max(dim=0)[0]
    return pooled
def manual_global_mean_pool(x, batch):
    batch_size = batch.max().item() + 1
    pooled = torch.zeros((batch_size, x.size(1)), device=x.device)
    count = torch.zeros(batch_size, device=x.device)
    for i in range(batch_size):
        pooled[i] = x[batch == i].sum(dim=0)
        count[i] = (batch == i).sum()
    pooled = pooled / count.unsqueeze(1)
    return pooled
class GCN(torch.nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, dropout, nlayer=2, res=True):
        super().__init__()
        self.args = args
        self.nhid = nhid
        self.res = res

        self.convs = torch.nn.ModuleList()
        nlast = nfeat
        for layer in range(nlayer):
            self.convs.append(GCNConv(nlast, nhid))
            nlast = nhid

        if res:
            self.lin = Linear(nhid*nlayer, nclass)
        else:
            self.lin = Linear(nhid, nclass)

        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):

        xs = self.embedding(x,edge_index,edge_weight,return_list=True)

        if self.res:
            x = torch.cat(xs, dim=-1)
        x = self.lin(x)

        return F.log_softmax(x, dim=1)

    
    def embedding(self, x, edge_index, edge_weight = None, return_list=False, return_logits=False):

        xs = []
        for gconv in self.convs:
            x = gconv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            if self.res:
                xs.append(x)

        if return_logits:
            if self.res:
                x = torch.cat(xs, dim=-1)
            y = self.lin(x)

            if return_list:
                return xs, F.log_softmax(y, dim=1)
            else:
                return x, F.log_softmax(y, dim=1)
        else:
            if return_list:
                return xs
            elif self.res:
                x = torch.cat(xs, dim=-1)

        return x


class SAGE(torch.nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, dropout, nlayer=2, res=True):
        super().__init__()
        self.args = args
        self.nhid = nhid
        self.res = res

        self.convs = torch.nn.ModuleList()
        nlast = nfeat
        for layer in range(nlayer):
            self.convs.append(SAGEConv(nlast, nhid))
            nlast = nhid

        if res:
            self.lin = Linear(nhid*nlayer, nclass)
        else:
            self.lin = Linear(nhid, nclass)

        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):

        xs = self.embedding(x,edge_index,edge_weight,return_list=True)

        if self.res:
            x = torch.cat(xs, dim=-1)
        x = self.lin(x)

        return F.log_softmax(x, dim=1)

    
    def embedding(self, x, edge_index, edge_weight = None, return_list=False, return_logits=False):

        xs = []
        for gconv in self.convs:
            x = gconv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            if self.res:
                xs.append(x)

        if return_logits:
            if self.res:
                x = torch.cat(xs, dim=-1)
            y = self.lin(x)

            if return_list:
                return xs, F.log_softmax(y, dim=1)
            else:
                return x, F.log_softmax(y, dim=1)
        else:
            if return_list:
                return xs
            elif self.res:
                x = torch.cat(xs, dim=-1)

        return x

class GIN(torch.nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, dropout, nlayer=2, res=True):
        super().__init__()
        self.args = args
        self.nhid = nhid
        self.res = res

        self.convs = torch.nn.ModuleList()
        nlast = nfeat
        for layer in range(nlayer):
            mlp = MLP(nlast, nhid, nhid, is_cls=False)
            self.convs.append(GINConv(mlp))
            nlast = nhid

        if res:
            self.lin = Linear(nhid*nlayer, nclass)
        else:
            self.lin = Linear(nhid, nclass)

        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):

        xs = self.embedding(x,edge_index,edge_weight,return_list=True)

        if self.res:
            x = torch.cat(xs, dim=-1)
        x = self.lin(x)

        return F.log_softmax(x, dim=1)

    
    def embedding(self, x, edge_index, edge_weight = None, return_list=False, return_logits=False):

        xs = []
        for gconv in self.convs:
            x = gconv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            if self.res:
                xs.append(x)

        if return_logits:
            if self.res:
                x = torch.cat(xs, dim=-1)
            y = self.lin(x)

            if return_list:
                return xs, F.log_softmax(y, dim=1)
            else:
                return x, F.log_softmax(y, dim=1)
        else:
            if return_list:
                return xs
            elif self.res:
                x = torch.cat(xs, dim=-1)

        return x

'''
class StructGNN(torch.nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, n_mem=8, dropout=0, nlayer=2, res=True, use_key=False, att='dp'):
        super().__init__()
        self.args = args
        self.nhid = nhid
        self.res = res

        self.pre_conv = nn.Linear(nfeat,nhid)
        self.convs = torch.nn.ModuleList()
        self.mem_cells = torch.nn.ModuleList()
        nlast = nhid
        for layer in range(nlayer):
            mlp = MLP(nlast*2, nhid, nhid, is_cls=False)
            self.convs.append(StructConv(mlp))
            
            mem = MemModule(nhid, k=n_mem, default=True, use_key=use_key, att=att)
            self.mem_cells.append(mem)

            nlast = nhid

        if res:
            self.lin = Linear(nhid*(nlayer+1), nclass)
            self.lin_weight = Linear(nhid*(nlayer+1), 1)
        else:
            self.lin = Linear(nhid, nclass)
            self.lin_weight = Linear(nhid*nlayer, 1)

        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):

        x = self.embedding(x,edge_index,edge_weight)
        x = self.lin(x)

        return F.log_softmax(x, dim=1)

    
    def embedding(self, x, edge_index, edge_weight = None, return_list=False, return_logits=False, return_sim=False):
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

            x, sim = mem_cell(x, return_s=True)
            sim_s.append(sim)

            if self.res:
                xs.append(x)

        if return_logits:
            if self.res:
                x = torch.cat(xs, dim=-1)
                x = torch.cat([pre_x,x], dim=-1)
            y = self.lin(x)

            if return_list:
                return xs, F.log_softmax(y, dim=1)
            else:
                return x, F.log_softmax(y, dim=1)
        else:
            if return_list:
                return xs
            elif self.res:
                x = torch.cat(xs, dim=-1)
                x = torch.cat([pre_x,x], dim=-1)

        if return_sim:
            return x, sim_s
        else:
            return x

    def predict_weight(self, x, edge_index, edge_weight=None, return_sim=False):
        if return_sim:
            x, sim_s = self.embedding(x,edge_index,edge_weight,return_sim=return_sim)
        else:
            x = self.embedding(x,edge_index,edge_weight,return_sim=return_sim)

        x = self.lin_weight(x)

        x = torch.sigmoid(x)
        norm = x.shape[0]/x.sum().detach()
        x = x*norm


        if return_sim:
            return x, sim_s
        else:
            return x
'''
class StructGNN(torch.nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, n_mem=[8,8], dropout=0, nlayer=2, res=True, use_key=False, att='dp'):
        super().__init__()
        self.args = args
        self.nhid = nhid
        self.res = res

        self.pre_conv = nn.Linear(nfeat,nhid)
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
            self.lin = Linear(nhid*(nlayer), nclass)
            self.lin_weight = Linear(nhid*(nlayer), 1)
        else:
            self.lin = Linear(nhid, nclass)
            self.lin_weight = Linear(nhid, 1)

        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):

        x = self.embedding(x,edge_index,edge_weight)
        x = self.lin(x)

        return F.log_softmax(x, dim=1)

    
    def embedding(self, x, edge_index, edge_weight = None, return_list=False, return_logits=False, return_sim=False):
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

            x, sim = mem_cell(x, return_s=True)
            sim_s.append(sim)

            if self.res:
                xs.append(x)

        if return_logits:
            if self.res:
                x = torch.cat(xs, dim=-1)
                #x = torch.cat([pre_x,x], dim=-1)
            y = self.lin(x)

            if return_list:
                return xs, F.log_softmax(y, dim=1)
            else:
                return x, F.log_softmax(y, dim=1)
        else:
            if return_list:
                return xs
            elif self.res:
                x = torch.cat(xs, dim=-1)
                #x = torch.cat([pre_x,x], dim=-1)

        if return_sim:
            return x, sim_s
        else:
            return x

    def predict_weight(self, x, edge_index, edge_weight=None, return_sim=False):
        if return_sim:
            x, sim_s = self.embedding(x,edge_index,edge_weight,return_sim=return_sim)
        else:
            x = self.embedding(x,edge_index,edge_weight,return_sim=return_sim)

        x = self.lin_weight(x)

        x = torch.sigmoid(x)
        norm = x.shape[0]/x.sum().detach()
        x = x*norm


        if return_sim:
            return x, sim_s
        else:
            return x


class GCNReweighter(torch.nn.Module):
    def __init__(self, args, nfeat, nhid, nclass,dropout=0, nlayer=2, res=True):
        # use a GCN for reweighting

        super().__init__()
        self.args = args
        self.nhid = nhid
        self.res = res

        self.convs = torch.nn.ModuleList()
        nlast = nfeat
        for layer in range(nlayer):
            self.convs.append(GCNConv(nlast, nhid))
            nlast = nhid

        if res:
            self.lin = Linear(nhid*(nlayer), nclass)
            self.lin_weight = Linear(nhid*(nlayer), 1)
        else:
            self.lin = Linear(nhid, nclass)
            self.lin_weight = Linear(nhid, 1)

        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):

        x = self.embedding(x,edge_index,edge_weight)
        x = self.lin(x)

        return F.log_softmax(x, dim=1)

    
    def embedding(self, x, edge_index, edge_weight = None, return_list=False, return_logits=False, return_sim=False):
        assert not (return_sim and return_list), "can not return both sim and list"
        assert not (return_sim and return_logits), "can not return both sim and logits"

        xs = []
        for gconv in self.convs:
            x = gconv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            if self.res:
                xs.append(x)

        if return_logits:
            if self.res:
                x = torch.cat(xs, dim=-1)
            y = self.lin(x)

            if return_list:
                return xs, F.log_softmax(y, dim=1)
            else:
                return x, F.log_softmax(y, dim=1)
        else:
            if return_list:
                return xs
            elif self.res:
                x = torch.cat(xs, dim=-1)

        if return_sim:
            return x, xs
        else:
            return x

    def predict_weight(self, x, edge_index, edge_weight=None, return_sim=False):
        if return_sim:
            x, sim_s = self.embedding(x,edge_index,edge_weight,return_sim=return_sim)
        else:
            x = self.embedding(x,edge_index,edge_weight,return_sim=return_sim)

        x = self.lin_weight(x)

        x = torch.sigmoid(x)
        norm = x.shape[0]/x.sum().detach()
        x = x*norm


        if return_sim:
            return x, sim_s
        else:
            return x

# ---------------------------------------------
# simple graph GNN model
# --------------------------------------------
class GraphGCN(torch.nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, dropout, nlayer=2, res=False):
        super().__init__()
        self.args = args
        self.nhid = nhid
        self.res = res

        self.convs = torch.nn.ModuleList()
        nlast = nfeat
        for layer in range(nlayer):
            self.convs.append(GCNConv(nlast, nhid))
            nlast = nhid

        if res:
            self.lin = Linear(nhid * nlayer * 2, nclass)
        else:
            self.lin = Linear(nhid * 2, nclass)

        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        if batch is None:  # No batch given
            print('no batch info given')
            import ipdb; ipdb.set_trace()
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



class GraphSAGE(torch.nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, dropout, nlayer=2, res=True):
        super().__init__()
        self.args = args
        self.nhid = nhid
        self.res = res

        self.convs = torch.nn.ModuleList()
        nlast = nfeat
        for layer in range(nlayer):
            self.convs.append(SAGEConv(nlast, nhid))
            nlast = nhid

        if res:
            self.lin = Linear(nhid*nlayer*2, nclass)
        else:
            self.lin = Linear(nhid*2, nclass)

        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        if batch is None: # No batch given
            print('no batch info given')
            ipdb.set_trace()
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


class GraphGIN(torch.nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, dropout, nlayer=2, res=True):
        super().__init__()
        self.args = args
        self.nhid = nhid
        self.res = res

        self.convs = torch.nn.ModuleList()
        nlast = nfeat
        for layer in range(nlayer):
            mlp = MLP(nlast, nhid, nhid, is_cls=False)
            self.convs.append(GINConv(mlp))
            nlast = nhid

        if res:
            self.lin = Linear(nhid*nlayer*2, nclass)
        else:
            self.lin = Linear(nhid*2, nclass)

        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        if batch is None: # No batch given
            print('no batch info given')
            ipdb.set_trace()
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
            ipdb.set_trace()
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


class StructATTGraphGNN(torch.nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, n_mem=[8,8], dropout=0, nlayer=2, res=True, use_key=False, att='dp'):
        super().__init__()
        assert nlayer==2, "StructATTGraphGNN is only implemented for 2 layers"
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
        
        gate_nn = MLP(nlast, nhid, 1, is_cls=False)
        up_nn = MLP(nlast, nhid, nhid*2, is_cls=False)
        self.att_1 = GlobalAttention(gate_nn=gate_nn, nn=up_nn)
        
        gate_nn = MLP(nlast, nhid, 1, is_cls=False)
        up_nn = MLP(nlast, nhid, nhid*2, is_cls=False)
        self.att_2 = GlobalAttention(gate_nn=gate_nn, nn=up_nn)

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
            ipdb.set_trace()
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

        graph_x = self.att_2(x, batch=batch)
        if self.res:
            graph_x1 = self.att_1(xs[0], batch=batch)
            graph_x = torch.cat((graph_x1,graph_x), dim=-1)

        if self.res:
            x = torch.cat(xs, dim=-1)
            #x = torch.cat([pre_x,x], dim=-1)    

        if graph_level:
            x = graph_x
            emb_list = [graph_x]
        else:
            emb_list = xs

        if return_logits:
            y = self.lin(graph_x)

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