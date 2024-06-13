 
import argparse
import os
import os.path as osp
import random

import GCL.augmentors as A
import GCL.losses as L
import numpy as np
import torch
from GCL.eval import get_split, SVMEvaluator
from GCL.models import DualBranchContrast
from torch.optim import Adam
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset, IMDB

from models.gin import GConv, Encoder
from models.utils import console_log
from prune.mask import Mask
import torch_geometric.graphgym.loader
from GCL.models.contrast_model import add_extra_mask
 
from GCL.losses import Loss

 
 
 
 
 
import numpy as np

def get_lord_error_fn(logits, Y, ord):
  errors = torch.nn.functional.softmax(logits, dim=1) - Y
  scores = np.linalg.norm(errors.detach().cpu().numpy(), ord=ord, axis=-1)
  return scores

def get_grad_norm_fn(loss):
  loss_grads = loss.grad
  scores = np.linalg.norm(loss_grads, axis=-1)
  return scores

class DualBranchContrast_diet(DualBranchContrast):
    def __init__(self, loss: Loss, mode: str, intraview_negs: bool = False, use_grad_norm: bool = False, ord: int = 1 ):
        super(DualBranchContrast_diet, self).__init__(loss=loss, mode=mode, intraview_negs=intraview_negs)
        self.use_grad_norm = use_grad_norm
        self.ord = ord


    def forward(self, h1=None, h2=None, g1=None, g2=None, batch=None, h3=None, h4=None,
                extra_pos_mask=None, extra_neg_mask=None):
        if self.mode == 'L2L':
            assert h1 is not None and h2 is not None
            anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=h1, sample=h2)
            anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=h2, sample=h1)
        elif self.mode == 'G2G':
            assert g1 is not None and g2 is not None
            anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=g1, sample=g2)
            anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=g2, sample=g1)
        else:   
            if batch is None or batch.max().item() + 1 <= 1:   
                assert all(v is not None for v in [h1, h2, g1, g2, h3, h4])
                anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=g1, sample=h2, neg_sample=h4)
                anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=g2, sample=h1, neg_sample=h3)
            else:   
                assert all(v is not None for v in [h1, h2, g1, g2, batch])
                anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=g1, sample=h2, batch=batch)
                anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=g2, sample=h1, batch=batch)

        pos_mask1, neg_mask1 = add_extra_mask(pos_mask1, neg_mask1, extra_pos_mask, extra_neg_mask)
        pos_mask2, neg_mask2 = add_extra_mask(pos_mask2, neg_mask2, extra_pos_mask, extra_neg_mask)
        l1 = self.loss(anchor=anchor1, sample=sample1, pos_mask=pos_mask1, neg_mask=neg_mask1, **self.kwargs)
        l2 = self.loss(anchor=anchor2, sample=sample2, pos_mask=pos_mask2, neg_mask=neg_mask2, **self.kwargs)
        if self.use_grad_norm:
            scores1 = get_grad_norm_fn(l1)
            scores2 = get_grad_norm_fn(l2)
        else:
            scores1 = get_lord_error_fn(anchor1, sample1, self.ord)
            scores2 = get_lord_error_fn(anchor2, sample2, self.ord)
        return (l1 + l2) * 0.5, scores1, scores2

 
def train(encoder_model, contrast_model, dataloader, optimizer, args):
    encoder_model.train()
    epoch_loss = 0

    pruneMask = Mask(encoder_model)
    prunePercent = args.prune_percent
    randomPrunePercent = args.random_prune_percent
    magnitudePrunePercent = prunePercent - randomPrunePercent
    pruneMask.magnitudePruning(magnitudePrunePercent, randomPrunePercent)

    for data in dataloader:
        data = data.to('cuda')
        optimizer.zero_grad()
         
        with torch.no_grad():
             
            encoder_model.set_prune_flag(True)
            features_2 = encoder_model(data.x, data.edge_index, data.batch)
             
        encoder_model.set_prune_flag(False)
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)

        _, _, _, _, g1, g2 = encoder_model(data.x, data.edge_index, data.batch)
        g1, g2 = [encoder_model.encoder.project(g) for g in [g1, g2]]
        loss, s1, s2 = contrast_model(g1=g1, g2=g2, batch=data.batch)
         
         
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss


def test(encoder_model, dataloader):
    encoder_model.eval()
    x = []
    y = []
    for data in dataloader:
        data = data.to('cuda')
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        _, g, _, _, _, _ = encoder_model(data.x, data.edge_index, data.batch)
        x.append(g)
        y.append(data.y)
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)

    split = get_split(num_samples=x.size()[0], train_ratio=0.8, test_ratio=0.1)
    result = SVMEvaluator(linear=True)(x, y, split)
    return result


def main():
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)
    device = torch.device('cuda')
    path = osp.join(osp.expanduser('~'), 'datasets/')
    print(path)
     
    dataset = TUDataset(path, name='PROTEINS')
    dataloader = DataLoader(dataset, batch_size=128)
    dataloader = list(dataloader)
    input_dim = max(dataset.num_features, 1)

    aug1 = A.Identity()
    aug2 = A.RandomChoice([A.RWSampling(num_seeds=1000, walk_length=10),
                           A.NodeDropping(pn=0.1),
                           A.FeatureMasking(pf=0.1),
                           A.EdgeRemoving(pe=0.1)], 1)
    gconv = GConv(input_dim=input_dim, hidden_dim=32, num_layers=2).to(device)
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2)).to(device)
    contrast_model = DualBranchContrast_diet(loss=L.InfoNCE(tau=0.2), mode='G2G').to(device)

    optimizer = Adam(encoder_model.parameters(), lr=0.01)

    for epoch in range(200):
        loss = train(encoder_model, contrast_model, dataloader, optimizer, args)
        print({'loss': loss})

    test_result = test(encoder_model, dataloader)
    print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--prune_percent', type=float, default=0.3, help="whole prune percentage")
    argparser.add_argument('--random_prune_percent', type=float, default=0.0, help="random prune percentage")
    args = argparser.parse_args()
    args.outdir = os.path.join('rpp{random_prune_percent}-pp{random_prune_percent}'.format(
        prune_percent=str(args.prune_percent), random_prune_percent=str(args.random_prune_percent)))

    if os.path.exists(args.outdir) is None:
        os.makedirs(os.path.join(args.outdir), exist_ok=True)

    console = console_log(args.outdir)
    main()