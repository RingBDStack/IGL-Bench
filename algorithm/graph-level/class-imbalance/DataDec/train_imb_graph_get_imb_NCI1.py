 
import argparse
import os
import os.path as osp
import random
import GCL.augmentors as A
import GCL.losses as L
import torch
from GCL.eval import get_split, SVMEvaluator
from torch.optim import Adam
from torch_geometric.data import DataLoader
import copy
from models.gin import GConv, Encoder, Encoder_self_damaged
from models.utils import console_log
from prune.mask import Mask
import numpy as np
from models.utils import get_lord_error_fn, get_grad_norm_fn, get_subset
from data_diet.utils import TUDataset_indices, DualBranchContrast_diet, static_sampler, cosine_schedule
from data_diet.dataset import shuffle, shuffle_for_ssl_return_train_index, get_TUDataset, TUDataset_indices_imb, shuffle_for_ssl_figure

 

def return_score(encoder_model, contrast_model,
          dataset,
          optimizer, args,
          sampled_list=None, unsampled_list=None,
          explore_rate=0.1, dataloader=None,
          data_prune_rate=0.5, is_error_rank=True):
     
    error_score_list = np.asarray([])
    grad_norm_list = np.asarray([])
    indices_list = np.asarray([])
    explore_num = int(explore_rate * len(dataset))
    diet_len = int(len(dataset) * data_prune_rate)

     
    encoder_model.train()
    epoch_loss = 0
    pruneMask = Mask(encoder_model)
    prunePercent = args.prune_percent
    randomPrunePercent = args.random_prune_percent
    magnitudePrunePercent = prunePercent - randomPrunePercent
    pruneMask.magnitudePruning(magnitudePrunePercent, randomPrunePercent)

     
    if sampled_list is None:
        sampled_list = np.asarray(list(range(len(dataset))))
    if unsampled_list is None:
        unsampled_list = np.asarray([]).astype(int)
     

     
    dataloader = list(dataloader)
     
    for (data, indices) in dataloader:
        data = data.to('cuda')
        optimizer.zero_grad()
         
        with torch.no_grad():
             
            encoder_model.eval()
            encoder_model.set_prune_flag(True)
            features_2 = encoder_model(data.x, data.edge_index, data.batch)
        encoder_model.train()
        encoder_model.set_prune_flag(True)
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)

        _, _, _, _, g1, g2 = encoder_model(data.x, data.edge_index, data.batch)
        encoder_model.set_prune_flag(False)
        g1, g2 = [encoder_model.encoder.project(g) for g in [g1, g2]]
        with torch.no_grad():
            loss, error_scores1, error_scores2, sample1, sample2 = contrast_model(g1=g1, g2=g2, batch=data.batch)
         

         
         
         
        error_scores = error_scores1 + error_scores2
        error_score_list = np.concatenate((error_score_list, error_scores))
         
        indices_list = np.concatenate((indices_list, indices.tolist()))
        optimizer.step()
        epoch_loss += loss.item()

     
    error_score_sorted_id = np.argsort(-error_score_list)
    grad_norm_sorted_id = np.argsort(-grad_norm_list)
    error_rank_id = indices_list[error_score_sorted_id]

     
    if is_error_rank:
        rank_id = error_rank_id.astype(int)
    else:
        grad_rank_id = indices_list[grad_norm_sorted_id]
        rank_id = grad_rank_id.astype(int)
    keep_num = diet_len - explore_num
    kept_sampled_list = rank_id[:keep_num]
    removed_sampled_list = rank_id[keep_num:]
    unsampled_list = np.concatenate((unsampled_list, removed_sampled_list))
    np.random.shuffle(unsampled_list)
    newly_added_sampled_list = unsampled_list[:explore_num]
    unsampled_list = unsampled_list[explore_num:]
    sampled_list = np.concatenate((kept_sampled_list, newly_added_sampled_list))

    return epoch_loss/len(dataloader), sampled_list[::-1].tolist(), unsampled_list, error_score_list[error_score_sorted_id]

def train(encoder_model, contrast_model,
          dataset,
          optimizer, args,
          sampled_list=None, unsampled_list=None,
          explore_rate=0.1, dataloader=None,
          data_prune_rate=0.5, is_error_rank=True):
     
    error_score_list = np.asarray([])
    grad_norm_list = np.asarray([])
    indices_list = np.asarray([])
    explore_num = int(explore_rate * len(dataset))
    diet_len = int(len(dataset) * data_prune_rate)

     
    encoder_model.train()
    epoch_loss = 0
    pruneMask = Mask(encoder_model)
    prunePercent = args.prune_percent
    randomPrunePercent = args.random_prune_percent
    magnitudePrunePercent = prunePercent - randomPrunePercent
    pruneMask.magnitudePruning(magnitudePrunePercent, randomPrunePercent)


     
    dataloader = list(dataloader)
    for (data, indices) in dataloader:
        data = data.to('cuda')
        optimizer.zero_grad()
         
        with torch.no_grad():
             
            encoder_model.eval()
            encoder_model.set_prune_flag(True)
            features_2 = encoder_model(data.x, data.edge_index, data.batch)
        encoder_model.train()
        encoder_model.set_prune_flag(True)
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)

        _, _, _, _, g1, g2 = encoder_model(data.x, data.edge_index, data.batch)
        encoder_model.set_prune_flag(False)

        g1, g2 = [encoder_model.encoder.project(g) for g in [g1, g2]]

        loss, error_scores1, error_scores2, sample1, sample2 = contrast_model(g1=g1, g2=g2, batch=data.batch)
        loss.backward()

        grad_norm1 = get_grad_norm_fn(sample1.grad.cpu().numpy())
        grad_norm2 = get_grad_norm_fn(sample2.grad.cpu().numpy())
        grad_norm = grad_norm1 + grad_norm2
        error_scores = error_scores1 + error_scores2
        error_score_list = np.concatenate((error_score_list, error_scores))
        grad_norm_list = np.concatenate((grad_norm_list, grad_norm))
        indices_list = np.concatenate((indices_list, indices.tolist()))
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss/len(dataloader), [], [], error_score_list


def test(encoder_model, dataloader):
    encoder_model.eval()
    x = []
    y = []
    for data, indices in dataloader:
        data = data.to('cuda')
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        with torch.no_grad():
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
    print('dataset path:', path)

     
    full_dataset, args.n_feat, args.n_class = get_TUDataset(args.dataset)

     
    train_val_index, val_index, test_index, \
    args.class_train_num_graph, args.class_val_num_graph = shuffle_for_ssl_return_train_index(
        args.dataset, full_dataset, args.imb_ratio, args.num_training, args.num_val)

    sampled_list = copy.deepcopy(train_val_index.tolist()) 
    sampled_list_full = copy.deepcopy(train_val_index.tolist()) 
    unsampled_list = None
     
     
     
    dataloader = DataLoader(full_dataset, batch_size=128,
                            sampler=static_sampler(sampled_list), drop_last=False, pin_memory=True)
     
     

    input_dim = max(args.n_feat, 1)  
    aug1 = A.Identity()
    aug2 = A.RandomChoice([A.RWSampling(num_seeds=1000, walk_length=10),
                           A.NodeDropping(pn=0.1),
                           A.FeatureMasking(pf=0.1),
                           A.EdgeRemoving(pe=0.1)], 1)
    gconv = GConv(input_dim=input_dim, hidden_dim=128, num_layers=2).to(device)
    encoder_model = Encoder_self_damaged(encoder=gconv, augmentor=(aug1, aug2)).to(device)
    contrast_model = DualBranchContrast_diet(loss=L.InfoNCE(tau=0.2), mode='G2G').to(device)

    optimizer = Adam(encoder_model.parameters(), lr=0.01)
    sampled_list_ep, unsampled_list_ep = copy.deepcopy(sampled_list), []
    indices_all_epoch = []
    error_score_list_all_epoch = []
    for epoch in range(args.num_epochs):
         
        indices_all_epoch.append(copy.deepcopy(sampled_list_full))
        data_prune_rate = 0.3 
        model_add_rate = cosine_schedule(epoch, args.num_epochs)
        args.prune_percent = args.random_prune_percent + (1 - model_add_rate) * (args.biggest_prune_percent - args.random_prune_percent)
        loss, sampled_list_ep, unsampled_list_ep, error_score_list = train(encoder_model, contrast_model,
                                                         train_val_index,  
                                                         optimizer, args,
                                                         sampled_list=np.asarray(sampled_list_ep).astype(int),
                                                         unsampled_list=np.asarray(unsampled_list_ep).astype(int),
                                                         explore_rate=args.explore_rate, dataloader=dataloader,
                                                         data_prune_rate=data_prune_rate, is_error_rank=args.is_error_rank)
         
        sampled_list.clear()
        sampled_list.extend(sampled_list_ep)
        print('%d ep'% epoch+'. prune rate %.2f:'%data_prune_rate+' loss %.2f'% loss)

     
    test_dataloader = DataLoader(full_dataset, batch_size=128, sampler=static_sampler(test_index), drop_last=False, pin_memory=True)
    test_dataloader = list(test_dataloader)
    test_result = test(encoder_model, test_dataloader)
    print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--biggest_prune_percent', type=float, default=0.75, help="whole prune percentage")
    argparser.add_argument('--prune_percent', type=float, default=0., help="whole prune percentage")
    argparser.add_argument('--random_prune_percent', type=float, default=0.25, help="random prune percentage")
    argparser.add_argument('--num_epochs', type=int, default=200, help="number of epochs")
    argparser.add_argument('--data_prune_rate', type=float, default=0.8, help="prune rate of data")
    argparser.add_argument('--explore_rate', type=float, default=0.0, help="explore rate")
    argparser.add_argument('--dataset', type=str, default='NCI1', help="dataset name")
    argparser.add_argument('--imb_ratio', type=float, default=0.9, help="imbalance ratio")
    argparser.add_argument('--is_error_rank', type=bool, default=False, help="use error rank or gradient rank")
    argparser.add_argument('--num_training', type=int, default=1000, help="number of training graphs")
    argparser.add_argument('--num_val', type=int, default=1000, help="number of validation graphs")

    args = argparser.parse_args()
    args.outdir = os.path.join('rpp{random_prune_percent}-pp{random_prune_percent}'.format(
        prune_percent=str(args.prune_percent), random_prune_percent=str(args.random_prune_percent)))

    if os.path.exists(args.outdir) is None:
        os.makedirs(os.path.join(args.outdir), exist_ok=True)

    console = console_log(args.outdir)
    main()