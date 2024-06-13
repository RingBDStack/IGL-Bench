import argparse
import torch
import os.path as osp
import GCL.losses as L
import GCL.augmentors as A
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.utils import to_networkx, from_networkx
from kornia.losses import FocalLoss
import random
from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split, LREvaluator
from GCL.eval.logistic_regression import LogisticRegression
from torch import nn
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score
from GCL.models import DualBranchContrast
from data_diet.utils import TUDataset_indices, DualBranchContrast_diet_grace, static_sampler, cosine_schedule, DualBranchContrast_diet_grace_node
from prune.mask import Mask
from torch_geometric.datasets import Planetoid
import statistics
from models.gin import make_gin_conv, make_gcn_conv
from models.utils import get_lord_error_fn, get_grad_norm_fn, get_subset
from ens_dataset.data_utils import get_dataset, split_semi_dataset, get_idx_info, make_longtailed_data_remove
import numpy as np
class LREvaluator_node(LREvaluator):
    def __init__(self, *args, **kwargs):
        super(LREvaluator_node, self).__init__(*args, **kwargs)
    def evaluate(self, x: torch.FloatTensor, y: torch.LongTensor, split: dict):
        device = x.device
        x = x.detach().to(device)
        input_dim = x.size()[1]
        y = y.to(device)
        num_classes = y.max().item() + 1
        classifier = LogisticRegression(input_dim, num_classes).to(device)
        optimizer = Adam(classifier.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        output_fn = nn.LogSoftmax(dim=-1)
         
        criterion = FocalLoss(alpha=0.5, gamma=2.0, reduction='mean')
        best_val_micro = 0
        best_test_micro = 0
        best_test_macro = 1
        best_epoch = 0

        with tqdm(total=self.num_epochs, desc='(LR)',
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]') as pbar:
            for epoch in range(self.num_epochs):
                classifier.train()
                optimizer.zero_grad()

                output = classifier(x[split['train']])
                loss = criterion(output_fn(output), y[split['train']])

                loss.backward()
                optimizer.step()

                if (epoch + 1) % self.test_interval == 0:
                    classifier.eval()
                    y_test = y[split['test']].detach().cpu().numpy()
                    y_pred = classifier(x[split['test']]).argmax(-1).detach().cpu().numpy()
                    test_micro = f1_score(y_test, y_pred, average='micro')
                    test_macro = f1_score(y_test, y_pred, average='macro')
                     
                    test_acc = accuracy_score(y_test, y_pred)
                     
                    test_bal_acc = balanced_accuracy_score(y_test, y_pred) 

                    y_val = y[split['valid']].detach().cpu().numpy()
                    y_pred = classifier(x[split['valid']]).argmax(-1).detach().cpu().numpy()
                    val_micro = f1_score(y_val, y_pred, average='micro')

                    if val_micro > best_val_micro:
                        best_val_micro = val_micro
                        best_test_micro = test_micro
                        best_test_macro = test_macro
                        best_epoch = epoch

                    pbar.set_postfix({'best test F1Mi': best_test_micro, 'F1Ma': best_test_macro})
                    pbar.update(self.test_interval)

        return {
            'micro_f1': best_test_micro,
            'macro_f1': best_val_micro,
            'test_acc': test_acc,
            'test_bal_acc': test_bal_acc,
        }


class GConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, activation, num_layers):
        super(GConv, self).__init__()
        self.activation = activation()
        self.layers = torch.nn.ModuleList()
        layer = make_gcn_conv(input_dim, hidden_dim)
        self.layers.append(layer)
        for _ in range(num_layers - 1):
            layer = make_gcn_conv(hidden_dim, hidden_dim)
            self.layers.append(layer)

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for i, conv in enumerate(self.layers):
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
        return z

class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor, hidden_dim, proj_dim):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor

        self.fc1 = torch.nn.Linear(hidden_dim, proj_dim)
        self.fc2 = torch.nn.Linear(proj_dim, hidden_dim)

    def forward(self, x, edge_index, edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)
        z = self.encoder(x, edge_index, edge_weight)
        z1 = self.encoder(x1, edge_index1, edge_weight1)
        z2 = self.encoder(x2, edge_index2, edge_weight2)
        return z, z1, z2

    def project(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)


 
def train(encoder_model, contrast_model, data, optimizer, sampled_list=None, unsampled_list=None, explore_rate=0.1, data_prune_rate=0.5):
     
    data_num = len(data.x)
    error_score_list = np.asarray([])
    grad_norm_list = np.asarray([])
    indices_list = np.asarray([])
    explore_num = int(explore_rate * data_num)
    diet_len = int(len(data.y) * data_prune_rate)

     
    if unsampled_list is None:
        unsampled_list = []
        data_pruned = data
         
        sampled_list = np.asarray(list(range(len(data.y))))
        unsampled_list = np.asarray(sampled_list[:len(data.y)//2])
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
        sampled_list = np.asarray(list(range(len(data.y))))
    else:
        if data.edge_attr is not None:
            data_networkx = to_networkx(data, node_attrs=['x'], edge_attrs=["edge_attrs"])
        else:
            data_networkx = to_networkx(data, node_attrs=['x'])
         
        remove_edges = []
        data_networkx.remove_nodes_from(unsampled_list)
        data_pruned = from_networkx(data_networkx)   
    if unsampled_list is None:
        unsampled_list = np.asarray([]).astype(int)
     
     
    encoder_model.train()
    epoch_loss = 0
    pruneMask = Mask(encoder_model)
    prunePercent = args.prune_percent
    randomPrunePercent = args.random_prune_percent
    magnitudePrunePercent = prunePercent - randomPrunePercent
    pruneMask.magnitudePruning(magnitudePrunePercent, randomPrunePercent)

    optimizer.zero_grad()
    data_pruned.to('cuda')
    z, z1, z2 = encoder_model(data_pruned.x, data_pruned.edge_index, data_pruned.edge_attr)
    h1, h2 = [encoder_model.project(x) for x in [z1, z2]]
    loss, error_scores1, error_scores2, sample1, sample2 = contrast_model(h1, h2)
    loss.backward()
    indices = np.arange(len(data_pruned.x))
    grad_norm1 = get_grad_norm_fn(sample1.grad.cpu().numpy())
    grad_norm2 = get_grad_norm_fn(sample2.grad.cpu().numpy())
    grad_norm = grad_norm1 
    error_scores = error_scores1 
    error_score_list = error_scores
    grad_norm_list = grad_norm
    indices_list = indices

    optimizer.step()

     
    error_score_sorted_id = np.argsort(-error_score_list)
    grad_norm_sorted_id = np.argsort(-grad_norm_list)
    error_rank_id, grad_rank_id = indices_list[error_score_sorted_id], indices_list[grad_norm_sorted_id]

    rand_id = error_rank_id   
    keep_num = diet_len - explore_num
    kept_sampled_list = rand_id[:keep_num] 
    removed_sampled_list = rand_id[keep_num:] 
    unsampled_list = np.concatenate((unsampled_list, removed_sampled_list))
    np.random.shuffle(unsampled_list)
    newly_added_sampled_list = unsampled_list[:explore_num]
    unsampled_list = unsampled_list[explore_num:]
    sampled_list = np.concatenate((sampled_list, newly_added_sampled_list))
    return loss.item(), sampled_list, unsampled_list


def test(encoder_model, data, data_train_mask, data_val_mask, data_test_mask):
    encoder_model.eval()
    with torch.no_grad():
        z, _, _ = encoder_model(data.x, data.edge_index, data.edge_attr)
    split = get_split_ens(num_samples=z.size()[0], data_train_mask=data_train_mask, data_val_mask=data_val_mask, data_test_mask=data_test_mask)
    result = LREvaluator_node()(z, data.y, split)
    return result

def get_split_ens(num_samples, data_train_mask, data_val_mask, data_test_mask):
    return {
        'train': data_train_mask,
        'valid': data_val_mask,
        'test': data_test_mask
    }

def main():
     
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)

    device = torch.device('cuda')

     
     
    assert args.imb_ratio == 100

     
    dataset = args.dataset
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', dataset)
    dataset = get_dataset(dataset, path, split_type='full')
    data = dataset[0]
    n_cls = data.y.max().item() + 1
    data = data.to(device)
    data_train_mask, data_val_mask, data_test_mask = data.train_mask.clone(), data.val_mask.clone(), data.test_mask.clone()
     
    stats = data.y[data_train_mask]
    n_data = []
    for i in range(n_cls):
        data_num = (stats == i).sum()
        n_data.append(int(data_num.item()))
    idx_info = get_idx_info(data.y, n_cls, data_train_mask)
    class_num_list = n_data

     
    class_num_list, data_train_mask, idx_info, train_node_mask, train_edge_mask = make_longtailed_data_remove(
        data.edge_index, \
        data.y, n_data, n_cls, args.imb_ratio, data_train_mask.clone())
     

    aug1 = A.Compose([A.EdgeRemoving(pe=0.3), A.FeatureMasking(pf=0.3)])
    aug2 = A.Compose([A.EdgeRemoving(pe=0.3), A.FeatureMasking(pf=0.3)])

    gconv = GConv(input_dim=dataset.num_features, hidden_dim=32, activation=torch.nn.ReLU, num_layers=2).to(device)
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2), hidden_dim=32, proj_dim=32).to(device)
    contrast_model = DualBranchContrast_diet_grace_node(loss=L.InfoNCE(tau=0.2), mode='L2L', intraview_negs=True).to(device)

    optimizer = Adam(encoder_model.parameters(), lr=0.01)
    sampled_list, unsampled_list = None, None
    for epoch in range(1, args.num_epochs):
        data_prune_rate = cosine_schedule(epoch, args.num_epochs)
        model_add_rate = cosine_schedule(epoch, args.num_epochs)
         
        args.prune_percent = args.random_prune_percent + (1 - model_add_rate) * (args.biggest_prune_percent - args.random_prune_percent)
        loss, sampled_list, unsampled_list = train(encoder_model, contrast_model, data, optimizer, sampled_list=sampled_list,
                     unsampled_list=unsampled_list,
                     explore_rate=0.1, data_prune_rate=data_prune_rate)
        real_prune_rate = len(sampled_list)/(len(sampled_list)+len(unsampled_list))
        print('%d ep' % epoch + '. data prune rate: %.2f' % real_prune_rate + '. model prune rate: %.2f' % args.prune_percent + '. loss %.2f' % loss)


    test_result = test(encoder_model, data, data_train_mask, data_val_mask, data_test_mask)
    print('test result: ', test_result)
    print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')
    print(f'(E): Best test test_acc={test_result["test_acc"]:.4f}, test_balanced_acc={test_result["test_bal_acc"]:.4f}')



if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--biggest_prune_percent', type=float, default=0.5, help="whole prune percentage")
    argparser.add_argument('--prune_percent', type=float, default=0.0, help="whole prune percentage")
    argparser.add_argument('--random_prune_percent', type=float, default=0.0, help="random prune percentage")
    argparser.add_argument('--num_epochs', type=int, default=1001, help="number of epochs")
    argparser.add_argument('--data_prune_rate', type=float, default=0.0, help="prune rate of data")
    argparser.add_argument('--dataset', type=str, default='Cora', help="dataset name, ['Cora', 'CiteSeer', 'PubMed']")
    argparser.add_argument('--imb_ratio', type=int, default=100, help="imbalance ratio")
     
    argparser.add_argument('--is_error_rank', type=bool, default=False, help="use error rank or gradient rank")
    argparser.add_argument('--num_training', type=int, default=120, help="number of training graphs")
    argparser.add_argument('--num_val', type=int, default=120, help="number of validation graphs")
    argparser.add_argument('--weight_decay', type=float, default=0.001, help="weight decay")

    args = argparser.parse_args()
    main()