import copy
import torch
import os.path as osp
import GCL.losses as L
import GCL.augmentors as A
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.utils import erdos_renyi_graph, to_networkx, from_networkx
from GCL.models import get_sampler
import numpy as np
from torch.optim import Adam
from GCL.eval import get_split, LREvaluator
from GCL.models import BootstrapContrast
from GCL.models.contrast_model import add_extra_mask
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import WikiCS
from models.utils import get_lord_error_fn, get_grad_norm_fn, get_subset
import networkx as nx

class Normalize(torch.nn.Module):
    def __init__(self, dim=None, norm='batch'):
        super().__init__()
        if dim is None or norm == 'none':
            self.norm = lambda x: x
        if norm == 'batch':
            self.norm = torch.nn.BatchNorm1d(dim)
        elif norm == 'layer':
            self.norm = torch.nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x)


class GConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.2,
                 encoder_norm='batch', projector_norm='batch'):
        super(GConv, self).__init__()
        self.activation = torch.nn.PReLU()
        self.dropout = dropout

        self.layers = torch.nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))

        self.batch_norm = Normalize(hidden_dim, norm=encoder_norm)
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            Normalize(hidden_dim, norm=projector_norm),
            torch.nn.PReLU(),
            torch.nn.Dropout(dropout))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for conv in self.layers:
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
            z = F.dropout(z, p=self.dropout, training=self.training)
        z = self.batch_norm(z)
        return z, self.projection_head(z)
 


class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor, hidden_dim, dropout=0.2, predictor_norm='batch'):
        super(Encoder, self).__init__()
        self.online_encoder = encoder
        self.target_encoder = None
        self.augmentor = augmentor
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            Normalize(hidden_dim, norm=predictor_norm),
            torch.nn.PReLU(),
            torch.nn.Dropout(dropout))

    def get_target_encoder(self):
        if self.target_encoder is None:
            self.target_encoder = copy.deepcopy(self.online_encoder)

            for p in self.target_encoder.parameters():
                p.requires_grad = False
        return self.target_encoder

    def update_target_encoder(self, momentum: float):
        for p, new_p in zip(self.get_target_encoder().parameters(), self.online_encoder.parameters()):
            next_p = momentum * p.data + (1 - momentum) * new_p.data
            p.data = next_p

    def forward(self, x, edge_index, edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)

        h1, h1_online = self.online_encoder(x1, edge_index1, edge_weight1)
        h2, h2_online = self.online_encoder(x2, edge_index2, edge_weight2)

        h1_pred = self.predictor(h1_online)
        h2_pred = self.predictor(h2_online)

        with torch.no_grad():
            _, h1_target = self.get_target_encoder()(x1, edge_index1, edge_weight1)
            _, h2_target = self.get_target_encoder()(x2, edge_index2, edge_weight2)

        return h1, h2, h1_pred, h2_pred, h1_target, h2_target


def train(encoder_model, contrast_model, data, optimizer, sampled_list=None, unsampled_list=None, explore_rate=0.1):
     
    data_num = len(data.x)
    error_score_list = []
    grad_norm_list = []
    indices_list = []
    error_score_list = np.asarray(error_score_list)
    grad_norm_list = np.asarray(grad_norm_list)
    indices_list = np.asarray(indices_list)
    explore_num = int(explore_rate * data_num)
    if unsampled_list is None:
        unsampled_list = []
        data_pruned = data
         
        sampled_list = list(range(len(data.y)))
        unsampled_list = sampled_list[:len(data.y)//2]
        if data.edge_attr is not None:
            data_networkx = to_networkx(data, node_attrs=['x'], edge_attrs=["edge_attrs"])
        else:
            data_networkx = to_networkx(data, node_attrs=['x'])

         
        remove_edges = []
        for node_i in unsampled_list:
            remove_edges.extend(data_networkx.edges(node_i))
        data_networkx.remove_edges_from(remove_edges)
        data_pruned = from_networkx(data_networkx) 
        data_pruned = data_pruned
     
    if sampled_list is None:
        data_pruned = data
        sampled_list = list(range(len(data.y)))
        unsampled_list = []
    else:
        sampled_list = list(range(len(data.y)))
         
        if data.edge_attr is not None:
            data_networkx = to_networkx(data, node_attrs=['x'], edge_attrs=["edge_attrs"])
        else:
            data_networkx = to_networkx(data, node_attrs=['x'])
         
        remove_edges = []
        for node_i in unsampled_list:
            remove_edges.extend(data_networkx.edges(node_i))
        data_networkx.remove_edges_from(remove_edges)
        data_pruned = from_networkx(data_networkx)   
        data_pruned = data_pruned

     

     
    encoder_model.train()
    optimizer.zero_grad()
     
    data_pruned = data_pruned.to('cuda:0')
    _, _, h1_pred, h2_pred, h1_target, h2_target = encoder_model(data_pruned.x, data_pruned.edge_index, data_pruned.edge_attr)
    loss, error_scores1, error_scores2, sample1, sample2 = contrast_model(h1_pred=h1_pred, h2_pred=h2_pred, h1_target=h1_target.detach(), h2_target=h2_target.detach())
    loss.backward()
    indices = np.arange(len(data.x))
    grad_norm1 = get_grad_norm_fn(sample1.grad.cpu().numpy())
    grad_norm2 = get_grad_norm_fn(sample2.grad.cpu().numpy())
    grad_norm = grad_norm1 + grad_norm2
    error_scores = error_scores1 + error_scores2
    error_score_list = np.concatenate((error_score_list, error_scores))
    grad_norm_list = np.concatenate((grad_norm_list, grad_norm))
    indices_list = np.concatenate((indices_list, indices.tolist()))

    optimizer.step()
    encoder_model.update_target_encoder(0.99)

     
    error_score_sorted_id = np.argsort(-error_score_list)
    grad_norm_sorted_id = np.argsort(-grad_norm_list)
    error_score_list = error_score_list[error_score_sorted_id]
    grad_norm_list = grad_norm_list[grad_norm_sorted_id]
    error_rank_id, grad_rank_id = indices_list[error_score_sorted_id], indices_list[grad_norm_sorted_id]

    print('error_rank_id:', error_rank_id)
    print('error_score_list:', error_score_list)
     
    rand_id = error_rank_id   
    keep_num = len(sampled_list) - explore_num
    kept_sampled_list = sampled_list[rand_id[:keep_num]]
    removed_sampled_list = unsampled_list[rand_id[keep_num:]]
    unsampled_list = np.concatenate((unsampled_list, removed_sampled_list))
    unsampled_list = np.random.shuffle(unsampled_list)
    newly_added_sampled_list = unsampled_list[:explore_num]
    unsampled_list = unsampled_list[explore_num:]
    sampled_list = np.concatenate((sampled_list, newly_added_sampled_list))
    return loss.item(), sampled_list, unsampled_list


def test(encoder_model, data):
    encoder_model.eval()
    h1, h2, _, _, _, _ = encoder_model(data.x, data.edge_index)
    z = torch.cat([h1, h2], dim=1)
    split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
    result = LREvaluator()(z, data.y, split)
    return result

class BootstrapContrast_diet(BootstrapContrast):
    def __init__(self, loss, mode='L2L', ord: int = 1, **kwargs):
        super(BootstrapContrast, self).__init__()
        self.loss = loss
        self.mode = mode
        self.sampler = get_sampler(mode, intraview_negs=False)
        self.ord = ord

    def forward(self, h1_pred=None, h2_pred=None, h1_target=None, h2_target=None,
                g1_pred=None, g2_pred=None, g1_target=None, g2_target=None,
                batch=None, extra_pos_mask=None):
        if self.mode == 'L2L':
            assert all(v is not None for v in [h1_pred, h2_pred, h1_target, h2_target])
            anchor1, sample1, pos_mask1, _ = self.sampler(anchor=h1_target, sample=h2_pred)
            anchor2, sample2, pos_mask2, _ = self.sampler(anchor=h2_target, sample=h1_pred)
        elif self.mode == 'G2G':
            assert all(v is not None for v in [g1_pred, g2_pred, g1_target, g2_target])
            anchor1, sample1, pos_mask1, _ = self.sampler(anchor=g1_target, sample=g2_pred)
            anchor2, sample2, pos_mask2, _ = self.sampler(anchor=g2_target, sample=g1_pred)
        else:
            assert all(v is not None for v in [h1_pred, h2_pred, g1_target, g2_target])
            if batch is None or batch.max().item() + 1 <= 1:   
                pos_mask1 = pos_mask2 = torch.ones([1, h1_pred.shape[0]], device=h1_pred.device)
                anchor1, sample1 = g1_target, h2_pred
                anchor2, sample2 = g2_target, h1_pred
            else:
                anchor1, sample1, pos_mask1, _ = self.sampler(anchor=g1_target, sample=h2_pred, batch=batch)
                anchor2, sample2, pos_mask2, _ = self.sampler(anchor=g2_target, sample=h1_pred, batch=batch)

        pos_mask1, _ = add_extra_mask(pos_mask1, extra_pos_mask=extra_pos_mask)
        pos_mask2, _ = add_extra_mask(pos_mask2, extra_pos_mask=extra_pos_mask)
        l1 = self.loss(anchor=anchor1, sample=sample1, pos_mask=pos_mask1)
        l2 = self.loss(anchor=anchor2, sample=sample2, pos_mask=pos_mask2)

        sample1.retain_grad()
        sample2.retain_grad()
        scores1 = get_lord_error_fn(anchor1, sample1, self.ord)
        scores2 = get_lord_error_fn(anchor2, sample2, self.ord)
        return (l1 + l2) * 0.5, scores1, scores2, sample1, sample2

def main():
    device = torch.device('cuda')
    path = osp.join(osp.expanduser('~'), 'datasets', 'WikiCS')
    dataset = WikiCS(path, transform=T.NormalizeFeatures())
    print('dataset path:', path)
    data = dataset[0].to(device)

    aug1 = A.Compose([A.EdgeRemoving(pe=0.5), A.FeatureMasking(pf=0.1)])
    aug2 = A.Compose([A.EdgeRemoving(pe=0.5), A.FeatureMasking(pf=0.1)])

    gconv = GConv(input_dim=dataset.num_features, hidden_dim=256, num_layers=2).to(device)
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2), hidden_dim=256).to(device)
    contrast_model = BootstrapContrast_diet(loss=L.BootstrapLatent(), mode='L2L').to(device)

    optimizer = Adam(encoder_model.parameters(), lr=0.01)

    sampled_list = None
    unsampled_list = None
    for epoch in range(1, 101):
        loss, sampled_list, unsampled_list = train(encoder_model, contrast_model, data, optimizer, sampled_list=sampled_list, unsampled_list=unsampled_list, explore_rate=0.1)
     
     
     
     
     
     
     

    test_result = test(encoder_model, data)
    print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')


if __name__ == '__main__':
    main()