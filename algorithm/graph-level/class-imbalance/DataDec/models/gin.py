import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GINConv, global_add_pool, GCNConv,SAGEConv
import torch.autograd as autograd
import math
class prune_linear(nn.Linear):
	def __init__(self, *args, **kwargs):
		super(prune_linear, self).__init__(*args, **kwargs)
		self.prune_mask = torch.ones(list(self.weight.shape))
		self.prune_flag = False

	def forward(self, input):
		if not self.prune_flag:
			weight = self.weight
		else:
			weight = self.weight * self.prune_mask
		return F.linear(input, weight, self.bias)
	
	def set_prune_flag(self, flag):
		self.prune_flag = flag

class prune_sage_conv(SAGEConv):
    def __init__(self, lin=None, input_dim=None, out_dim=None, *args, **kwargs):
        super(prune_sage_conv, self).__init__(in_channels=input_dim, out_channels=out_dim, *args, **kwargs)
        self.prune_flag = False
        self.lin = lin

    def set_prune_flag(self, flag):
        self.prune_flag = flag

def make_sage_conv(input_dim, out_dim):
    mlp = prune_linear(input_dim, out_dim)
    sage_conv = prune_sage_conv(mlp,input_dim, out_dim)
    return sage_conv

class prune_gin_conv(GINConv):
	def __init__(self, *args, **kwargs):
		super(prune_gin_conv, self).__init__(*args, **kwargs)
		self.prune_flag = False
	
	def set_prune_flag(self, flag):
		self.prune_flag = flag


def make_gin_conv(input_dim, out_dim):
	 
	mlp = prune_linear(input_dim, out_dim)
	gin_conv = prune_gin_conv(mlp, train_eps=True)

	return gin_conv


class prune_gcn_conv(GCNConv):
	def __init__(self, lin=None, input_dim=None, out_dim=None, *args, **kwargs):
		 
		super(prune_gcn_conv, self).__init__(in_channels=input_dim, out_channels=out_dim,*args, **kwargs)
		self.prune_flag = False
		self.lin = lin

	def set_prune_flag(self, flag):
		self.prune_flag = flag

def make_gcn_conv(input_dim, out_dim):
	mlp = prune_linear(input_dim, out_dim)
	gcn_conv = prune_gcn_conv(mlp, input_dim, out_dim)

	return gcn_conv

class subnet_linear(nn.Linear):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
		nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

		self.use_subset = True

		self.score_mask = None

	def set_prune_rate(self, prune_rate):
		self.prune_rate = prune_rate

	def init_weight_with_score(self, prune_rate):
		self.weight.data = self.weight.data * GetSubnet.apply(self.clamped_scores, prune_rate).data
		self.use_subset = False

	@property
	def clamped_scores(self):
		return self.scores.abs()

	def get_subnet(self):
		return GetSubnet.apply(self.clamped_scores, self.prune_rate).detach()

	def discard_low_score(self, discard_rate):
		self.score_mask = GetSubnet.apply(self.clamped_scores, 1 - discard_rate).detach() == 0
		self.scores[self.score_mask].data.zero_()
		self.scores.grad[self.score_mask] = 0

	def clear_low_score_grad(self):
		if self.score_mask is not None:
			self.scores.grad[self.score_mask] = 0

	def clear_subset_grad(self):
		subset = self.get_subnet()
		mask = subset == 1
		self.weight.grad[mask] = 0

	def lr_scale_zero(self, lr_scale):
		subset = self.get_subnet()
		mask = subset == 0
		self.weight.grad[mask].data *= lr_scale

	def weight_decay_custom(self, weight_decay, weight_decay_on_zero):
		subset = self.get_subnet()
		mask = subset == 1

		l2_reg_subset = torch.norm(self.weight[mask])
		l2_reg_zero = torch.norm(self.weight[~mask])

		loss = weight_decay * l2_reg_subset + weight_decay_on_zero * l2_reg_zero
		loss.backward()

	def forward(self, x):
		if self.use_subset:
			subnet = GetSubnet.apply(self.clamped_scores, self.prune_rate)
			w = self.weight * subnet
		else:
			w = self.weight

		x = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

		return x


class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
         
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

         
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out

    @staticmethod
    def backward(ctx, g):
         
        return g, None

class GConv(nn.Module):
	def __init__(self, input_dim, hidden_dim, num_layers):
		super(GConv, self).__init__()
		self.layers = nn.ModuleList()
		self.batch_norms = nn.ModuleList()
		self.prune_flag = False
		for i in range(num_layers):
			if i == 0:
				layer = make_gin_conv(input_dim, hidden_dim)
				self.layers.append(layer)
			else:
				layer = make_gin_conv(hidden_dim, hidden_dim)
				self.layers.append(layer)
			self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
		
		project_dim = hidden_dim * num_layers
		self.project = torch.nn.Sequential(
			nn.Linear(project_dim, project_dim),
			nn.ReLU(inplace=True),
			nn.Linear(project_dim, project_dim))
	
	def set_prune_flag(self, flag):
		self.prune_flag = flag
		for gin_i in self.layers:
			gin_i.set_prune_flag(flag)
	
	def forward(self, x, edge_index, batch):
		z = x
		zs = []
		for conv, bn in zip(self.layers, self.batch_norms):
			z = conv(z, edge_index)
			z = F.relu(z)
			z = bn(z)
			zs.append(z)
		gs = [global_add_pool(z, batch) for z in zs]
		z, g = [torch.cat(x, dim=1) for x in [zs, gs]]
		return z, g

class GCNV(nn.Module):
	def __init__(self, input_dim, hidden_dim, num_layers):
		super(GCNV, self).__init__()
		self.layers = nn.ModuleList()
		self.batch_norms = nn.ModuleList()
		self.prune_flag = False
		for i in range(num_layers):
			if i == 0:
				layer = make_gcn_conv(input_dim, hidden_dim)
				self.layers.append(layer)
			else:
				layer = make_gcn_conv(hidden_dim, hidden_dim)
				self.layers.append(layer)
			self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
		
		project_dim = hidden_dim * num_layers
		self.project = torch.nn.Sequential(
			nn.Linear(project_dim, project_dim),
			nn.ReLU(inplace=True),
			nn.Linear(project_dim, project_dim))
	
	def set_prune_flag(self, flag):
		self.prune_flag = flag
		for gin_i in self.layers:
			gin_i.set_prune_flag(flag)
	
	def forward(self, x, edge_index, batch):
		z = x
		zs = []
		for conv, bn in zip(self.layers, self.batch_norms):
			z = conv(z, edge_index)
			z = F.relu(z)
			z = bn(z)
			zs.append(z)
		gs = [global_add_pool(z, batch) for z in zs]
		z, g = [torch.cat(x, dim=1) for x in [zs, gs]]
		return z, g

class SAGE(nn.Module):
	def __init__(self, input_dim, hidden_dim, num_layers):
		super(SAGE, self).__init__()
		self.layers = nn.ModuleList()
		self.batch_norms = nn.ModuleList()
		self.prune_flag = False
		for i in range(num_layers):
			if i == 0:
				layer = make_sage_conv(input_dim, hidden_dim)
				self.layers.append(layer)
			else:
				layer = make_sage_conv(hidden_dim, hidden_dim)
				self.layers.append(layer)
			self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
		
		project_dim = hidden_dim * num_layers
		self.project = torch.nn.Sequential(
			nn.Linear(project_dim, project_dim),
			nn.ReLU(inplace=True),
			nn.Linear(project_dim, project_dim))
	
	def set_prune_flag(self, flag):
		self.prune_flag = flag
		for gin_i in self.layers:
			gin_i.set_prune_flag(flag)
	
	def forward(self, x, edge_index, batch):
		z = x
		zs = []
		for conv, bn in zip(self.layers, self.batch_norms):
			z = conv(z, edge_index)
			z = F.relu(z)
			z = bn(z)
			zs.append(z)
		gs = [global_add_pool(z, batch) for z in zs]
		z, g = [torch.cat(x, dim=1) for x in [zs, gs]]
		return z, g

class Encoder(torch.nn.Module):
	def __init__(self, encoder, augmentor):
		super(Encoder, self).__init__()
		self.encoder = encoder
		self.augmentor = augmentor
	
	def forward(self, x, edge_index, batch):
		aug1, aug2 = self.augmentor
		x1, edge_index1, edge_weight1 = aug1(x, edge_index)
		x2, edge_index2, edge_weight2 = aug2(x, edge_index)
		z, g = self.encoder(x, edge_index, batch)
		z1, g1 = self.encoder(x1, edge_index1, batch)
		z2, g2 = self.encoder(x2, edge_index2, batch)
		return z, g, z1, z2, g1, g2
	
	def set_prune_flag(self, flag):
		self.prune_flag = flag
		self.encoder.set_prune_flag(flag)


class Encoder_self_damaged(torch.nn.Module):
	def __init__(self, encoder, augmentor):
		super(Encoder_self_damaged, self).__init__()
		self.encoder = encoder
		self.augmentor = augmentor

	def forward(self, x, edge_index, batch):
		aug1, aug2 = self.augmentor
		x1, edge_index1, edge_weight1 = aug1(x, edge_index)
		x2, edge_index2, edge_weight2 = aug2(x, edge_index)
		z, g = self.encoder(x, edge_index, batch)
		z1, g1 = self.encoder(x1, edge_index1, batch)
		self.set_prune_flag(self.prune_flag)
		z2, g2 = self.encoder(x2, edge_index2, batch)
		self.set_prune_flag(False)
		return z, g, z1, z2, g1, g2

	def set_prune_flag(self, flag):
		self.prune_flag = flag
		self.encoder.set_prune_flag(flag)