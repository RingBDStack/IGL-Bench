"""
Our code is based on GraphENS:
https://github.com/JoonHyung-Park/GraphENS
"""

import os.path as osp
import os
import random
import torch
import torch.nn.functional as F
from nets import *
from data_utils import *
from args import parse_args,save_to_yaml
from models import *
from losses import *
from sklearn.metrics import balanced_accuracy_score, f1_score,roc_auc_score,accuracy_score
from sklearn.preprocessing import label_binarize
import statistics
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Arg Parser ##
args = parse_args()
save_to_yaml(args)


## Handling exception from arguments ##
assert not (args.warmup < 1 and args.tam)
# assert args.imb_ratio > 1

## Load Dataset ##
dataset = args.dataset
path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', dataset)
dataset = get_dataset(dataset, path, split_type='public')
data = dataset[0]
n_cls = data.y.max().item() + 1
data = data.to(device)
def get_split(all_idx, all_label, nclass):
    total_nodes = len(all_idx)
    train_each = (total_nodes * 0.1) // nclass 
    valid_each = (total_nodes * 0.1) // nclass 

    # Initialize node lists for each class
    train_list = [0 for _ in range(nclass)]
    train_node = [[] for _ in range(nclass)]
    train_idx  = []
    
    # Create train mask
    train_mask = np.zeros(total_nodes, dtype=bool)

    # Assign nodes to train set
    for idx in all_idx:
        label = all_label[idx]
        if train_list[label] < train_each:
            train_list[label] += 1
            train_node[label].append(idx)
            train_idx.append(idx)
            train_mask[idx] = True
        if sum(train_list) == train_each * nclass:
            break

    assert sum(train_list) == train_each * nclass, "Training split does not match the expected size."

    # Create validation and test masks
    valid_mask = np.zeros(total_nodes, dtype=bool)
    test_mask = np.zeros(total_nodes, dtype=bool)

    # Prepare remaining indices for validation and test sets
    remaining_idx = list(set(all_idx) - set(train_idx))
    valid_list = [0 for _ in range(nclass)]
    valid_idx  = []

    # Assign nodes to validation set
    for idx in remaining_idx:
        label = all_label[idx]
        if valid_list[label] < valid_each:
            valid_list[label] += 1
            valid_idx.append(idx)
            valid_mask[idx] = True
        if sum(valid_list) == valid_each * nclass:
            break

    assert sum(valid_list) == valid_each * nclass, "Validation split does not match the expected size."

    # Assign remaining nodes to test set
    test_idx = list(set(remaining_idx) - set(valid_idx))
    for idx in test_idx:
        test_mask[idx] = True

    return train_mask, valid_mask, test_mask
def get_split_arxiv(all_idx, all_label, nclass):
    num_samples = len(all_idx)
    train_each = int(num_samples * 0.1 / nclass) 
    valid_each = int(num_samples * 0.1 / nclass) 

    all_label = np.squeeze(all_label)
    label_counts = np.bincount(all_label, minlength=nclass) 

    train_list = [0] * nclass
    valid_list = [0] * nclass

    train_class = [min(train_each, int(count * 0.1)) for count in label_counts]
    valid_class = [min(valid_each, int(count * 0.1)) for count in label_counts]

    train_mask = np.zeros(num_samples, dtype=bool)
    valid_mask = np.zeros(num_samples, dtype=bool)
    test_mask = np.zeros(num_samples, dtype=bool)

    after_train_idx = []

    for idx in all_idx:
        label = all_label[idx]
        if train_list[label] < train_class[label]:
            train_list[label] += 1
            train_mask[idx] = True
        else:
            after_train_idx.append(idx)

    for idx in after_train_idx:
        label = all_label[idx]
        if valid_list[label] < valid_class[label]:
            valid_list[label] += 1
            valid_mask[idx] = True
        else:
            test_mask[idx] = True
    
    return train_mask, valid_mask, test_mask

def train():
    global class_num_list, aggregator
    global data_train_mask, data_val_mask, data_test_mask

    model.train()
    optimizer.zero_grad()        

    output = model(data.x, data.edge_index)

    ## Apply TAM ##
    output = adjust_output(args, output, data.edge_index, data.y, \
        data_train_mask, aggregator, class_num_list, epoch)
        
    criterion(output, data.y[data_train_mask]).backward()

    with torch.no_grad():
        model.eval()
        output = model(data.x, data.edge_index)
        if data.y.dim()==2:
            val_loss= F.cross_entropy(output[data_val_mask], data.y[data_val_mask].squeeze())
        else:
            val_loss= F.cross_entropy(output[data_val_mask], data.y[data_val_mask])

    optimizer.step()
    scheduler.step(val_loss)


@torch.no_grad()
def test():
    model.eval()
    logits = model(data.x, data.edge_index)
    accs, baccs, f1s, aurocs = [], [], [], []

    n_classes = logits.size(1) 

    softmax = torch.nn.Softmax(dim=1)
    
    for i, mask in enumerate([data_train_mask, data_val_mask, data_test_mask]):
        probabilities = softmax(logits[mask]).cpu().numpy()
        y_true = data.y[mask].cpu().numpy()

        y_true_bin = label_binarize(y_true, classes=range(n_classes))

        pred = logits[mask].max(1)[1]
        y_pred = pred.cpu().numpy()
        acc = accuracy_score(y_true, y_pred)
        bacc = balanced_accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')

        auroc_per_class = [roc_auc_score(y_true_bin[:, c], probabilities[:, c]) for c in range(n_classes)]
        auroc_avg = np.mean(auroc_per_class)
        aurocs.append(auroc_avg)

        accs.append(acc)
        baccs.append(bacc)
        f1s.append(f1)

    return accs, baccs, f1s, aurocs


## Log for Experiment Setting ##
setting_log = "Dataset: {}, ratio: {}, net: {}, n_layer: {}, feat_dim: {}, tam: {}".format(
    args.dataset, str(args.imb_ratio), args.net, str(args.n_layer), str(args.feat_dim), str(args.tam))

repeatition = 10
seed = 0
avg_val_acc_f1, avg_test_acc, avg_test_bacc, avg_test_f1,avg_test_auroc = [], [], [], [], []
for r in range(repeatition):

    ## Fix seed ##
    torch.cuda.empty_cache()
    seed += 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #random.seed(seed)
    np.random.seed(seed)
    mask_list = [i for i in range(data.num_nodes)]
    random.seed(args.shuffle_seed)
    random.shuffle(mask_list)
    labels = data.y.to('cpu')
    if args.dataset =='arxiv':
        data_train_mask, data_val_mask, data_test_mask = get_split_arxiv(mask_list,labels.numpy(),n_cls)
    else:
        data_train_mask, data_val_mask, data_test_mask = get_split(mask_list,labels.numpy(),n_cls)
    #if args.dataset in ['squirrel', 'chameleon', 'Wisconsin']:
    #    data_train_mask, data_val_mask, data_test_mask = data.train_mask[:,r%10].clone(), data.val_mask[:,r%10].clone(), data.test_mask[:,r%10].clone()
    #else:
        #data_train_mask, data_val_mask, data_test_mask = data.train_mask.clone(), data.val_mask.clone(), data.test_mask.clone()
    data_train_mask = torch.from_numpy(data_train_mask)
    ## Data statistic ##
    stats = data.y[data_train_mask]
    n_data = []
    for i in range(n_cls):
        data_num = (stats == i).sum()
        n_data.append(int(data_num.item()))
    data_train_mask = data_train_mask.to('cuda')
    if args.dataset =='arxiv':
        idx_info = get_idx_info_arxiv(data.y, n_cls, data_train_mask)
    else:
        idx_info = get_idx_info(data.y, n_cls, data_train_mask)
    class_num_list = n_data

    # for artificial imbalanced setting: only the last imb_class_num classes are imbalanced
    imb_class_num = n_cls // 2
    new_class_num_list = []
    max_num = np.max(class_num_list[:n_cls-imb_class_num])
    for i in range(n_cls):
        if args.imb_ratio > 1 and i > n_cls-1-imb_class_num: #only imbalance the last classes
            new_class_num_list.append(min(int(max_num*(1./args.imb_ratio)), class_num_list[i]))
        else:
            new_class_num_list.append(class_num_list[i])
    class_num_list = new_class_num_list

    if args.imb_ratio > 1:
        data_train_mask, idx_info = split_semi_dataset(len(data.x), n_data, n_cls, class_num_list, idx_info, data.x.device)

    ## Model Selection ##
    if args.net == 'GCN':
        model = create_gcn(nfeat=dataset.num_features, nhid=args.feat_dim,
                        nclass=n_cls, dropout=0.5, nlayer=args.n_layer)
    elif args.net == 'GAT':
        model = create_gat(nfeat=dataset.num_features, nhid=args.feat_dim,
                        nclass=n_cls, dropout=0.5, nlayer=args.n_layer)
    elif args.net == "SAGE":
        model = create_sage(nfeat=dataset.num_features, nhid=args.feat_dim,
                        nclass=n_cls, dropout=0.5, nlayer=args.n_layer)
    else:
        raise NotImplementedError("Not Implemented Architecture!")

    ## Criterion Selection ##
    if args.loss_type == 'ce': # CE
        criterion = CrossEntropy()
    elif args.loss_type == 'bs':
        criterion = BalancedSoftmax(class_num_list)
    else:
        raise NotImplementedError("Not Implemented Loss!")

    model = model.to(device)
    criterion = criterion.to(device)

    # Set optimizer
    optimizer = torch.optim.Adam([
        dict(params=model.reg_params, weight_decay=5e-4),
        dict(params=model.non_reg_params, weight_decay=0),], lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor = 0.5,
                                                           patience = 100,
                                                           verbose=False)

    # Train models
    best_val_acc_f1 = 0
    aggregator = MeanAggregation()
    for epoch in range(1, 501):

        train()
        accs, baccs, f1s,aurocs = test()
        train_acc, val_acc, tmp_test_acc = accs
        train_f1, val_f1, tmp_test_f1 = f1s
        val_acc_f1 = (val_acc + val_f1) / 2.
        if val_acc_f1 > best_val_acc_f1:
            best_val_acc_f1 = val_acc_f1
            test_acc = accs[2]
            test_bacc = baccs[2]
            test_f1 = f1s[2]
            test_auroc = aurocs[2]

    avg_val_acc_f1.append(best_val_acc_f1)
    avg_test_acc.append(test_acc)
    avg_test_bacc.append(test_bacc)
    avg_test_f1.append(test_f1)
    avg_test_auroc.append(test_auroc)

## Calculate statistics ##
acc_CI =  (statistics.stdev(avg_test_acc) / (repeatition ** (1/2)))
bacc_CI =  (statistics.stdev(avg_test_bacc) / (repeatition ** (1/2)))
f1_CI =  (statistics.stdev(avg_test_f1) / (repeatition ** (1/2)))
auroc_CI =  (statistics.stdev(avg_test_auroc) / (repeatition ** (1/2)))
avg_acc = statistics.mean(avg_test_acc)
avg_auroc = statistics.mean(avg_test_auroc)
avg_bacc = statistics.mean(avg_test_bacc)
avg_f1 = statistics.mean(avg_test_f1)
avg_val_acc_f1 = statistics.mean(avg_val_acc_f1)

avg_log = 'Test Acc: {:.4f} +- {:.4f}, BAcc: {:.4f} +- {:.4f}, F1: {:.4f} +- {:.4f}, AUROC : {:.4f} +- {:.4f}'
avg_log = avg_log.format(avg_acc ,acc_CI ,avg_bacc, bacc_CI, avg_f1, f1_CI, avg_auroc, auroc_CI)
log = "{}\n{}".format(setting_log, avg_log)
print(log)
avg_log = "seed: {}\n{}".format(args.shuffle_seed, avg_log)
os.makedirs('res', exist_ok=True)
file_path = os.path.join('res', f"{args.dataset}_{args.net}.txt")
with open(file_path, 'a') as file:
    file.write(avg_log + '\n')
