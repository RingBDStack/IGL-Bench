from typing import Optional, Tuple

import numpy as np
import torch
import os
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, fill_diag, matmul, mul
from torch_sparse import sum as sparsesum
import torch.nn.functional as F

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch.nn import Sequential as seq, Parameter,LeakyReLU,init,Linear
from torch_geometric.utils import add_self_loops, remove_self_loops,degree,softmax


@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, flow="source_to_target", dtype=None):
    # type: (Tensor, OptTensor, Optional[int], bool, bool, str, Optional[int]) -> PairTensor  # noqa
    pass


@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, flow="source_to_target", dtype=None):
    # type: (SparseTensor, OptTensor, Optional[int], bool, bool, str, Optional[int]) -> SparseTensor  # noqa
    pass


def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, flow="source_to_target", dtype=None):

    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        assert flow in ["source_to_target"]
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sparsesum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    else:
        assert flow in ["source_to_target", "target_to_source"]
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        idx = col if flow == "source_to_target" else row
        deg = scatter_add(edge_weight, idx, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


class GCNConv(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
    The adjacency matrix can include other values than :obj:`1` representing
    edge weights via the optional :obj:`edge_weight` tensor.

    Its node-wise formulation is given by:

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}^{\top} \sum_{j \in
        \mathcal{N}(v) \cup \{ i \}} \frac{e_{j,i}}{\sqrt{\hat{d}_j
        \hat{d}_i}} \mathbf{x}_j

    with :math:`\hat{d}_i = 1 + \sum_{j \in \mathcal{N}(i)} e_{j,i}`, where
    :math:`e_{j,i}` denotes the edge weight from source node :obj:`j` to target
    node :obj:`i` (default: :obj:`1.0`)

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        normalize (bool, optional): Whether to add self-loops and compute
            symmetric normalization coefficients on the fly.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge weights :math:`(|\mathcal{E}|)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})`
    """

    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, in_channels: int, out_channels: int,
                 k_ricci,e_poinc,n_components,n_components_p,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.k_ricci = k_ricci
        self.e_poinc = e_poinc

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.lin = Linear(in_channels, out_channels, bias=False)

        widths=[n_components,out_channels]
        widths_p=[n_components_p,out_channels]
        self.hmpnn=create_wmlp(widths,out_channels,1)
        self.ham=create_wmlp(widths_p,out_channels,1)
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj, alpha_hp: float,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache
            edge_weight = edge_weight.view(-1, 1)
        
        x = self.lin(x)

        edge_weight=self.hmpnn(self.k_ricci)
        edge_weight=softmax(edge_weight,edge_index[0])

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)
        p_weight=self.ham(self.e_poinc)
        p_weight=F.leaky_relu(p_weight)

        if self.bias is not None:
            out += self.bias

        return out+alpha_hp*p_weight

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight * x_j

    def update(self, aggr_out):
        return aggr_out
        
    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)


def create_wmlp(widths,nfeato,lbias):
    mlp_modules=[]
    for k in range(len(widths)-1):
        mlp_modules.append(Linear(widths[k],widths[k+1],bias=False))
        mlp_modules.append(LeakyReLU(0.2,True))
    mlp_modules.append(Linear(widths[len(widths)-1],nfeato,bias=lbias))
    return seq(*mlp_modules)

class Net(torch.nn.Module):
    def __init__(self,data,num_features,num_hidden,num_classes,k_ricci,e_poinc,n_components,n_components_p):
        super(Net, self).__init__()
        self.conv1 = GCNConv(num_features, num_hidden,k_ricci,e_poinc,n_components,n_components_p, cached=True)
        self.conv2 = GCNConv(num_hidden, num_classes, k_ricci,e_poinc,n_components,n_components_p, cached=True)

    def forward(self,data,alpha):
        x = F.dropout(data.x,p=0.6,training=self.training)
        x = self.conv1(x, data.edge_index, alpha)
        x = F.elu(x)
        x = F.dropout(x,p=0.6,training=self.training)
        x = self.conv2(x, data.edge_index, alpha)
        return x

def num(strings):
    try:
        return int(strings)
    except ValueError:
        return float(strings)

def set_model(data, config, device= 'cuda'):
    #ricci
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_dir = os.path.join(current_dir, '../../..')
    name = data.data_name
    ricci_file = os.path.join(file_dir,f'hyperemb/{data.data_name}.edge_list')
    f=open(ricci_file)
    cur_list=list(f)
    if name=='Cora' or name == 'Actor' or name=='Chameleon' or name=='Squirrel' or name == 'Computers'or name=='PubMed':
        ricci_cur=[[] for i in range(len(cur_list))]
        for i in range(len(cur_list)):
            ricci_cur[i]=[num(s) for s in cur_list[i].split(' ',2)]
    else:
        ricci_cur=[[] for i in range(2*len(cur_list))]
        for i in range(len(cur_list)):
            ricci_cur[i]=[num(s) for s in cur_list[i].split(' ',2)]
            ricci_cur[i+len(cur_list)]=[ricci_cur[i][1],ricci_cur[i][0],ricci_cur[i][2]]
    ricci_cur=sorted(ricci_cur)
    k_ricci=[i[2] for i in ricci_cur]
    k_ricci=k_ricci+[0 for i in range(data.x.size(0))]
    k_ricci=torch.tensor(k_ricci, dtype=torch.float)
    data.k_ricci=k_ricci.view(-1,1)
    data.n_components=1
    #poincare
    data.edge_index, _ = remove_self_loops(data.edge_index)
    
    keys_file = os.path.join(f'hyperemb/{data.data_name}_keys.npy')
    values_file = os.path.join(f'hyperemb/{data.data_name}_values.npy')
    keys=np.load(keys_file)
    values=np.load(values_file)
    
    e_poinc = dict(zip(keys, values))
    data.n_components_p = values.shape[1]
    alls = dict(enumerate(np.ones((data.num_nodes,data.n_components_p)), 0))
    alls.update(e_poinc)
    e_poinc = torch.tensor(np.array([alls[i] for i in alls]))
    data.e_poinc = e_poinc.to(torch.float32)
    data.edge_index, _ = add_self_loops(data.edge_index,num_nodes=data.x.size(0))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data.k_ricci = data.k_ricci.to(device)
    data.e_poinc = data.e_poinc.to(device)
    
    num_classes = data.y.cpu().numpy().max().item() + 1
    model= Net(data,data.num_features, config.hidden_dim, num_classes, data.k_ricci,
               data.e_poinc,data.n_components,data.n_components_p).to(device)
    
    data = data.to(device)
    return model, data