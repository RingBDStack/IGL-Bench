import numpy as np
import torch
from torch.nn import Sequential as seq, Parameter,LeakyReLU,init,Linear

from typing import List, Optional, Tuple, Union

import torch.nn.functional as F
from torch import Tensor
from torch.nn import LSTM
from torch_sparse import SparseTensor, matmul

from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, Size
from torch_geometric.utils import add_self_loops, remove_self_loops,degree,softmax

class SAGEConv(MessagePassing):
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W}_2 \cdot
        \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j

    If :obj:`project = True`, then :math:`\mathbf{x}_j` will first get
    projected via

    .. math::
        \mathbf{x}_j \leftarrow \sigma ( \mathbf{W}_3 \mathbf{x}_j +
        \mathbf{b})

    as described in Eq. (3) of the paper.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        aggr (string or Aggregation, optional): The aggregation scheme to use.
            Any aggregation of :obj:`torch_geometric.nn.aggr` can be used,
            *e.g.*, :obj:`"mean"`, :obj:`"max"`, or :obj:`"lstm"`.
            (default: :obj:`"mean"`)
        normalize (bool, optional): If set to :obj:`True`, output features
            will be :math:`\ell_2`-normalized, *i.e.*,
            :math:`\frac{\mathbf{x}^{\prime}_i}
            {\| \mathbf{x}^{\prime}_i \|_2}`.
            (default: :obj:`False`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        project (bool, optional): If set to :obj:`True`, the layer will apply a
            linear transformation followed by an activation function before
            aggregation (as described in Eq. (3) of the paper).
            (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **inputs:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **outputs:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V_t}|, F_{out})` if bipartite
    """
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        k_ricci,e_poinc,n_components,n_components_p,
        aggr: Optional[Union[str, List[str], Aggregation]] = "mean",
        normalize: bool = False,
        root_weight: bool = True,
        project: bool = False,
        bias: bool = True,
        **kwargs,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight
        self.project = project
        self.k_ricci = k_ricci
        self.e_poinc = e_poinc
        
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        if aggr == 'lstm':
            kwargs.setdefault('aggr_kwargs', {})
            kwargs['aggr_kwargs'].setdefault('in_channels', in_channels[0])
            kwargs['aggr_kwargs'].setdefault('out_channels', in_channels[0])

        super().__init__(aggr, **kwargs)
        widths=[n_components,out_channels]
        widths_p=[n_components_p,out_channels]
        self.hmpnn=create_wmlp(widths,in_channels[0],1)
        self.ham=create_wmlp(widths_p,out_channels,1)

        if self.project:
            self.lin = Linear(in_channels[0], in_channels[0], bias=True)

        if self.aggr is None:
            self.fuse = False  # No "fused" message_and_aggregate.
            self.lstm = LSTM(in_channels[0], in_channels[0], batch_first=True)

        if isinstance(self.aggr_module, MultiAggregation):
            aggr_out_channels = self.aggr_module.get_out_channels(
                in_channels[0])
        else:
            aggr_out_channels = in_channels[0]

        self.lin_l = Linear(aggr_out_channels, out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        if self.project:
            self.lin.reset_parameters()
        self.aggr_module.reset_parameters()
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, alpha_hp: float,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        if self.project and hasattr(self, 'lin'):
            x = (self.lin(x[0]).relu(), x[1])

        # propagate_type: (x: OptPairTensor)
        out_weight=self.hmpnn(self.k_ricci)
        out_weight=softmax(out_weight,edge_index[0])
        out = self.propagate(x=x,edge_index=edge_index,out_weight=out_weight)
        out = self.lin_l(out)
        p_weight=self.ham(self.e_poinc)
        p_weight=F.leaky_relu(p_weight)
        out = out+alpha_hp*p_weight
        x_r = x[1]
        if self.root_weight and x_r is not None:
            out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor, out_weight: Tensor) -> Tensor:
        return out_weight*x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, aggr={self.aggr})')


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
        self.conv1 = SAGEConv(num_features, num_hidden,k_ricci,e_poinc,n_components,n_components_p)
        self.conv2 = SAGEConv(num_hidden, num_classes,k_ricci,e_poinc,n_components,n_components_p)

    def forward(self, data, alpha):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.relu(self.conv1(x, edge_index, alpha))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, alpha)
        return F.log_softmax(x, dim=1)

def num(strings):
    try:
        return int(strings)
    except ValueError:
        return float(strings)

def call(data,name,num_features,num_classes,num_hidden):
    #ricci
    filename='hyperemb/'+name+'.edge_list'
    f=open(filename)
    cur_list=list(f)
    if name=='Cora' or name == 'Actor' or name=='chameleon' or name=='squirrel'or name=='computers'or name=='pubmed':
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
    keys=np.load('hyperemb/'+name+'_keys.npy')
    values=np.load('hyperemb/'+name+'_values.npy')
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

    data = data.to(device)
    model= Net(data,num_features,num_hidden,num_classes,data.k_ricci,data.e_poinc,data.n_components,data.n_components_p).to(device)
    return model, data



