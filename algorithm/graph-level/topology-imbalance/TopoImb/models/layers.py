from typing import Callable, Optional, Union

import torch
from torch import Tensor
from torch_sparse import SparseTensor, matmul

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch_geometric.nn.inits import reset as reset
from torch_geometric.nn import global_mean_pool, global_max_pool
import ipdb
import torch.nn
import math
import torch.nn.functional as F

class StructConv(MessagePassing):
    r"""Adapted from GIN, following torch_geometric


    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """
    def __init__(self, nn: Callable,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)

        x_r = x[1]
        if x_r is not None:
            #out += (1 + self.eps) * x_r
            out = torch.cat((out,x_r),dim=-1)

        return self.nn(out)


    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        ipdb.set_trace()
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'

class MemModule(torch.nn.Module):
    r'''
    

    Args:
        in_feat (int): dimension of input feature
        k (int): number of informative templates
        default (binary): whether use default template
        thresh (float): similarity to the default template
        use_key (binary): whether use key_memory
        att (str): 'dp' or 'mlp'
    '''

    def __init__(self, in_feat, k=8, default=True, thresh=0.2, use_key=False, att='dp'):
        super().__init__()

        self.in_feat = in_feat
        self.default = default
        self.thresh = thresh
        if self.default:
            self.k = k+1
        else:
            self.k = k

        self.value_memory = torch.nn.Parameter(torch.zeros((in_feat, self.k)))

        self.use_key = use_key
        if self.use_key:
            self.key_memory = torch.nn.Parameter(torch.zeros((in_feat, self.k)))
    
        self.att = att
        if self.att == 'mlp':
            self.att_model = MLP(in_feat*2, in_feat, 1, is_cls=False)

        self.reset_parameters()


    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.value_memory, a=math.sqrt(5))
        if self.use_key:
            torch.nn.init.kaiming_uniform_(self.key_memory, a=math.sqrt(5))

        return 

    def forward(self, feature, return_s=False):

        if self.use_key:
            key_tensor = self.key_memory
        else:
            key_tensor = self.value_memory

        if self.att == 'dp':
            s_matrix = torch.mm(feature, key_tensor) #not recommended
            s_matrix = torch.sigmoid(s_matrix)

        elif self.att == 'mlp':
            N = feature.shape[0]
            a_input = torch.cat([feature.repeat(1, self.k).view(N * self.k, -1), key_tensor.t().repeat(N, 1)], dim=1).view(N, -1, 2 * self.in_feat)
            s_matrix = self.att_model(a_input).squeeze()
            s_matrix = torch.sigmoid(s_matrix)
        
        s_matrix_mod = s_matrix.clone()
        s_matrix_mod[:,-1] = self.thresh
        s_matrix_mod = torch.softmax(s_matrix_mod,dim=-1)
        output = torch.mm(s_matrix_mod, self.value_memory.t())

        if not return_s:
            return output
        else:
            return output, s_matrix_mod


class MLP(torch.nn.Module):
    def __init__(self, in_feat, hidden_size, out_size, layers=2, dropout=0.1, is_cls=True):
        super(MLP, self).__init__()

        modules = []
        in_size = in_feat
        for layer in range(layers-1):
            modules.append(torch.nn.Linear(in_size, hidden_size))
            in_size = hidden_size
            modules.append(torch.nn.LeakyReLU(0.1))

        self.model = torch.nn.Sequential(*modules)
        self.cls = torch.nn.Linear(in_size, out_size)

        self.is_cls = is_cls

    def forward(self, features):
        output = self.embedding(features)
        output = self.cls(output)

        if self.is_cls:
            return F.log_softmax(output, dim=1)
        else:
            return output

    def embedding(self, features):
        output = self.model(features)

        return output