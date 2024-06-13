# from matplotlib.pyplot import plot
import csv
import time

import torch
import numpy as np
import random
import os
# import datasets as datasets
from torch_geometric.loader import DataLoader, RandomNodeSampler
import models.wl_models as wl_models
import models.models as models
import utils
import trainers.trainer as trainer
from utils import load_split, cal_imbalance_ratio
from trainers.ReweighterTrainer import ReweighterTrainer
from trainers.ReweightGraphTrainer import ReweighterGraphTrainer
from trainers.SSLTrainer_node import TopoNodeTrainer
from trainers.SSLTrainer_graph import TopoGraphTrainer
import math
# import torch_geometric.datasets as tg_dataset
from torch_geometric.data import InMemoryDataset
import copy
# from plots import plot_tsne,plot_dist1D,plot_chart, plot_single_grid
# from tensorboardX import SummaryWriter
# import ipdb
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from dataset import *
import torch_geometric.transforms as T

# from collections import Counter
'''
###-------------------------------
initialization, load dataset
###------------------------------
'''
metric = {}

metric['acc'] = []
metric['macro_F'] = []
metric['bacc'] = []
metric['auroc'] = []
metric['pretrain_time'] = []
metric['train_time'] = []
metric['total_time'] = []
metric['silhouette'] = []
###configure arguments
args = utils.get_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device('cuda' if args.cuda else 'cpu')

dataset, args.nfeat, args.n_class = get_TUDataset(args.dataset, pre_transform=T.ToSparseTensor(remove_edge_index=False))
args.nclass = args.n_class
# print('nclass:', args.nclass)
# print('n_class:', args.n_class)
# print('num_features:', args.nfeat)
graph_num = len(dataset)
labels = [data.y.item() for data in dataset]
unique_labels, counts = np.unique(labels, return_counts=True)
# print(unique_labels)
n_data = counts

args.num_train = (int)(len(dataset) * 0.1)
args.num_val = (int)(len(dataset) * 0.1 / args.n_class)

args.c_train_num, args.c_val_num = get_class_num(
    args.imb_ratio, args.num_train, args.num_val, args.dataset, args.n_class, n_data)
args.y = torch.tensor([data.y.item() for data in dataset], dtype=torch.int32)

del labels, n_data
large_dataset_list = ['DD', 'REDDIT-BINARY', 'COLLAB', 'FRANKENSTEIN']
multi_class_dataset_list = ['COLLAB']
if args.dataset in large_dataset_list:
    args.use_early_stop = True
else:
    args.use_early_stop = False
if args.dataset in multi_class_dataset_list:
    args.is_multi_class = True
else:
    args.is_multi_class = False
train_mask, val_mask, test_mask, boundary_size = load_split(
    load_path=osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'TU', args.dataset),
    split_mode=args.split_mode)

train_dataset = dataset[train_mask]
val_dataset = dataset[val_mask]
test_dataset = dataset[test_mask]
head_avg, tail_avg, imbalance_ratio = cal_imbalance_ratio(train_dataset, boundary_size)
print(args)

for count in range(5):
    random.seed(args.seed + count)
    torch.manual_seed(args.seed + count)
    np.random.seed(args.seed + count)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed + count)
        torch.cuda.manual_seed_all(args.seed + count)
        torch.backends.cudnn.deterministic = True

    # train_mask, val_mask, test_mask = shuffle(dataset, args.c_train_num, args.c_val_num, args.y)

    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    valoader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    testloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    # raise NotImplementedError("graph-type data is not implemented yet")

    '''
    ###------------------------------------------------------------------------------
    go through WL algorithm in torch.tensor: 
    output:
        topo_labels, torch.tensor
        topo_size_dict, dict of int for size of each topology group 
    ###------------------------------------------------------------------------------
    '''

    # get topo labels with wl-algorithm
    if args.datatype == 'node':
        data = dataset[0]
        steps = math.ceil((len(dataset) / args.batch_size))
        topo_path = 'topo_file/{}_{}.npy'.format(args.topo_initial, args.dataset)
        if os.path.exists(topo_path):
            topo_labels_np = np.load(topo_path)
            topo_labels = torch.tensor(topo_labels_np, dtype=torch.long, device=device)
        else:
            # initialize node label before wl algorithm, as clus
            if args.topo_initial == 'label':
                # initialize topo label as class label
                clus = dataset[0].y.cpu().numpy()
            elif args.topo_initial == 'emb':
                MLP_emb = dataset[0].x.cpu().numpy()
                pca = PCA(n_components=8)
                trans_x = pca.fit_transform(MLP_emb)
                scale = 100 / (trans_x.max() + 0.000000000000001)
                trans_x = trans_x * scale
                print('start clustering')
                clus = utils.clust(trans_x, n_clusters=6)
                print('finish clustering')

            # run wl algorithm
            datas = dataset[0]
            datas.x = dataset[0].x.new(clus).reshape(-1).long()

            wlmodel = wl_models.WL_model(args, nfeat=args.nfeat,
                                         nhid=args.nhid,
                                         nclass=args.nclass,
                                         dropout=0,
                                         nlayer=args.nlayer).to(device)
            WLtrainer = trainer.ClsTrainer(args, wlmodel, dataset=dataset)
            for epoch in range(10):
                # train
                if True:
                    data = copy.deepcopy(datas)
                    log_info = WLtrainer.train_step(data.to(device), epoch)
                    print('train log at  epoch{}: {}'.format(epoch, log_info))

            # get topo
            data = copy.deepcopy(datas)
            data = data.to(device)
            wlmodel = wlmodel.eval()
            topo_labels = wlmodel.wl(data.x, data.edge_index).detach()

            np.save(topo_path, topo_labels.cpu().numpy())

    elif args.datatype == 'graph':
        allloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        steps = math.ceil((len(dataset) / args.batch_size))
        topo_path = 'topo_file/{}_{}.npy'.format(args.topo_initial, args.dataset)
        for batch, data in enumerate(allloader):
            print('loaded data')
        if os.path.exists(topo_path):
            topo_labels_np = np.load(topo_path)
            topo_labels = torch.tensor(topo_labels_np, dtype=torch.long, device=device)
        else:
            # for graph dataset, use argmax in x as initial node clus

            wlmodel = wl_models.WLGraph_model(args, nfeat=args.nfeat,
                                              nhid=args.nhid,
                                              nclass=args.nclass,
                                              dropout=0,
                                              nlayer=args.nlayer)
            wlmodel = wlmodel.to(device)
            WLtrainer = trainer.GClsTrainer(args, wlmodel, dataset=dataset)

            ###train
            for epoch in range(1):
                # train
                for batch, data in enumerate(allloader):
                    log_info = WLtrainer.train_step(data.to(device), epoch)
                    print('train log at  epoch{}: {}'.format(epoch, log_info))

            # get topo dist tensor
            for batch, data in enumerate(allloader):
                wlmodel = wlmodel.eval()
                graph_wl_tensor = wlmodel.graph_wl_dist(data.x, data.edge_index, data.batch).detach()

            # clustering
            graph_clust = utils.clust(graph_wl_tensor.cpu().numpy(), n_clusters=16)
            topo_labels = torch.tensor(graph_clust, device=device)

            np.save(topo_path, topo_labels.cpu().numpy())

    # check if ground-truth topo_labels are provided
    topo_label_ori = topo_labels
    # if 'topo_label' in data.keys():
    #     topo_labels = data.topo_label.to(device)
    # else:
    topo_labels = topo_labels
    # get topo_size dict
    topo_size_dict = {}
    for topo in set(topo_labels.cpu().numpy()):
        topo_size = (topo_labels == topo).sum()
        topo_size_dict[topo] = topo_size.item()

    '''
    ###-------------------------------
    initialize/load GNN model
    ###------------------------------
    '''
    class_weights = None
    if args.datatype == 'node':
        if args.model == 'gcn':
            model = models.GCN(args, nfeat=args.nfeat,
                               nhid=args.nhid,
                               nclass=args.nclass,
                               dropout=args.dropout,
                               nlayer=args.nlayer)
        if args.model == 'sage':
            model = models.SAGE(args, nfeat=args.nfeat,
                                nhid=args.nhid,
                                nclass=args.nclass,
                                dropout=args.dropout,
                                nlayer=args.nlayer)
        if args.model == 'gin':
            model = models.GIN(args, nfeat=args.nfeat,
                               nhid=args.nhid,
                               nclass=args.n_class,
                               dropout=args.dropout,
                               nlayer=args.nlayer)

        if args.reweighter == 'struct':
            reweighter = models.StructGNN(args, nfeat=args.nfeat,
                                          nhid=args.nhid,
                                          nclass=args.nclass,
                                          dropout=0,
                                          n_mem=args.n_mem,
                                          nlayer=2,
                                          use_key=args.use_key,
                                          att=args.att)
        elif args.reweighter == 'class':
            reweighter = None
            class_weights = utils.class_weight(dataset)
        elif args.reweighter == 'gcn':
            reweighter = models.GCNReweighter(args, nfeat=args.nfeat, nhid=args.nhid, nclass=args.nclass, dropout=0)

    elif args.datatype == 'graph':
        if args.model == 'gcn':
            model = models.GraphGCN(args, nfeat=args.nfeat,
                                    nhid=args.nhid,
                                    nclass=args.nclass,
                                    dropout=args.dropout,
                                    nlayer=args.nlayer)
        elif args.model == 'sage':
            model = models.GraphSAGE(args, nfeat=args.nfeat,
                                     nhid=args.nhid,
                                     nclass=args.nclass,
                                     dropout=args.dropout,
                                     nlayer=args.nlayer)
        elif args.model == 'gin':
            model = models.GraphGIN(args, nfeat=args.nfeat,
                                    nhid=args.nhid,
                                    nclass=args.n_class,
                                    dropout=args.dropout,
                                    nlayer=args.nlayer,
                                    res=args.res)

        if args.reweighter == 'struct':
            reweighter = models.StructGraphGNN(args, nfeat=args.nfeat,
                                               nhid=args.nhid,
                                               nclass=args.nclass,
                                               dropout=0,
                                               n_mem=args.n_mem,
                                               nlayer=args.nlayer,
                                               use_key=args.use_key,
                                               att=args.att)
        elif args.reweighter == 'structATT':
            reweighter = models.StructATTGraphGNN(args, nfeat=args.nfeat,
                                                  nhid=args.nhid,
                                                  nclass=args.nclass,
                                                  dropout=0,
                                                  n_mem=args.n_mem,
                                                  nlayer=2,
                                                  use_key=args.use_key,
                                                  att=args.att)
        elif args.reweighter == 'class':
            reweighter = None
        elif args.reweighter == 'gcn':
            reweighter = models.GCNReweighter()

    if args.load is not None:
        model = utils.load_model(args, model, name='model_{}'.format(args.load))
        reweighter = utils.load_model(args, reweighter, name='reweighter_{}'.format(args.load))

    model = model.to(device)
    if reweighter is not None:
        reweighter = reweighter.to(device)

    '''
    ###-------------------------------
    train: model
    ###------------------------------
    '''
    start_time = time.time()
    ###set trainers
    Node_Retrainer_dict = {'cls': trainer.ClsTrainer, 'wlcls': TopoNodeTrainer, }
    Graph_Retrainer_dict = {'cls': trainer.GClsTrainer, 'wlcls': TopoGraphTrainer, }

    trainers = []
    if args.datatype == 'node':
        DOWNtrainer = ReweighterTrainer(args, model, reweighter=reweighter, weight=args.reweight_weight,
                                        dataset=dataset, weights=class_weights)
        Retrainer_dict = Node_Retrainer_dict
    elif args.datatype == 'graph':
        DOWNtrainer = ReweighterGraphTrainer(args, model, reweighter=reweighter, weight=args.reweight_weight,
                                             dataset=dataset)
        Retrainer_dict = Graph_Retrainer_dict
    trainers.append(DOWNtrainer)

    re_trainers = []  # auxiliary tasks for reweighter
    if args.reweight_task is not None:
        for re_task in args.reweight_task:
            if re_task == 'wlcls':
                retrainer = Retrainer_dict[re_task](args, reweighter, dataset=dataset,
                                                    out_size=len(set(topo_label_ori.cpu().numpy())),
                                                    shared_encoder=True)
            else:
                retrainer = Retrainer_dict[re_task](args, reweighter, dataset=dataset)
            re_trainers.append(retrainer)

    ###pretrain reweighter
    if args.pretrain_reweighter and len(re_trainers) != 0:
        for epoch in range(args.epochs):
            if epoch % args.test_interval == 0:
                for re_task, re_trainer in zip(args.reweight_task, re_trainers):
                    # test all data inside dataset
                    log_test = {}
                    data_iter = 0
                    for batch, data in enumerate(testloader):
                        if re_task == 'wlcls':
                            if args.datatype == 'node':
                                topo_label_test = topo_label_ori
                            elif args.datatype == 'graph':
                                topo_label_test = topo_label_ori[-int(graph_num * args.test_ratio):]
                                topo_label_test = topo_label_test[data_iter:data_iter + data.y.shape[0]]
                                data_iter = data_iter + data.y.shape[0]

                            log_info = re_trainer.test(data.to(device), label=topo_label_test)
                        else:
                            log_info = re_trainer.test(data.to(device))
                        for key in log_info:
                            if key not in log_test.keys():
                                log_test[key] = utils.meters(orders=1)
                            log_test[key].update(log_info[key], 1)
                    if args.save and epoch % args.test_interval == 0:
                        utils.save_model(args, reweighter, epoch=epoch, name='reweighter')
                        if log_info['acc_test'] > best_test_acc:
                            best_test_acc = log_info['acc_test']
                            utils.save_model(args, reweighter, name='best')

            for re_task, re_trainer in zip(args.reweight_task, re_trainers):
                data_iter = 0
                for batch, data in enumerate(dataloader):
                    if re_task == 'wlcls':
                        if args.datatype == 'node':
                            topo_label_train = topo_label_ori
                        elif args.datatype == 'graph':
                            topo_label_train = topo_label_ori[:int(graph_num * args.sup_ratio)]
                            topo_label_train = topo_label_train[data_iter:data_iter + data.y.shape[0]]
                            data_iter = data_iter + data.y.shape[0]
                        log_info = re_trainer.train_step(data.to(device), epoch, label=topo_label_train)
                    else:
                        log_info = re_trainer.train_step(data.to(device), epoch)
                    print('reweighter train log at {} in epoch{}: {}'.format(batch, epoch, log_info))

        pretrain_iter = epoch * steps + batch + 2
    else:
        pretrain_iter = 0
    mid_time = time.time()
    ###train model
    steps = math.ceil((len(dataset) / args.batch_size))
    best_test_acc = 0
    best_val_acc = 0
    silhouette_avg = 0.0
    patience = 0
    best_test_log = {}
    for epoch in range(args.epochs):
        # train model
        for _trainer in trainers:  #
            data_iter = 0
            for batch, data in enumerate(dataloader):
                log_info = _trainer.train_step(data.to(device), epoch)
                # print('train log at {} in epoch{}: {}'.format(batch, epoch, log_info))

                # train reweighter
                if len(re_trainers) != 0:
                    for re_task, re_trainer in zip(args.reweight_task, re_trainers):
                        if re_task == 'wlcls':
                            if args.datatype == 'node':
                                topo_label_train = topo_label_ori
                            elif args.datatype == 'graph':
                                # compare topo_labels[:int(graph_num*args.sup_ratio)][data_iter:data_iter+data.y.shape[0]] with data.topo_label can validate the correctness
                                topo_label_train = topo_label_ori[train_mask]
                                topo_label_train = topo_label_train[data_iter:data_iter + data.y.shape[0]]
                                data_iter = data_iter + data.y.shape[0]

                            log_info = re_trainer.train_step(data.to(device), epoch, label=topo_label_train)
                        else:
                            log_info = re_trainer.train_step(data.to(device), epoch)
                        # print('reweighter train log at {} in epoch{}: {}'.format(batch, epoch, log_info))

            if 'struct' in args.reweighter:
                for batch, data in enumerate(dataloader):
                    log_info = _trainer.train_mem_step(data.to(device), epoch)
                    # print('reweighter mem train log at {} in epoch{}: {}'.format(batch, epoch, log_info))
                    '''
                    for key in log_info:
                        if args.log:
                            writer.add_scalar('reweighter_mem_'+key, log_info[key], epoch*steps+batch+1)
                    '''
        # val
        for ctrainer in trainers:
            # test all data inside dataset
            val_log_test = {}
            val_log_info = ctrainer.test(valoader, device)
            for key in val_log_info:
                if key not in val_log_test.keys():
                    val_log_test[key] = utils.meters(orders=1)
                val_log_test[key].update(val_log_info[key], 1)
            print('Val log in step{}: {}'.format(epoch * steps, {key: val_log_test[key].avg() for key in val_log_info}))
        if val_log_info['acc_test'] >= best_val_acc:
            best_val_acc = val_log_info['acc_test']
            for ctrainer in trainers:
                # test all data inside dataset
                log_test = {}
                log_info = ctrainer.test(testloader, device)
                for key in log_info:
                    if key not in log_test.keys():
                        log_test[key] = utils.meters(orders=1)
                    log_test[key].update(log_info[key], 1)

                print(
                    'test log in step{}: {}'.format(epoch * steps, {key: log_test[key].avg() for key in log_info}))
                best_test_log = log_info
                if log_info['acc_test'] > best_test_acc:
                    best_test_acc = log_info['acc_test']

                # save model
                if args.save and epoch % args.test_interval == 0:
                    utils.save_model(args, model, epoch=epoch)
                    utils.save_model(args, reweighter, epoch=epoch, name='reweighter')
                    if log_info['acc_test'] > best_test_acc:
                        utils.save_model(args, model, name='best')


        else:
            patience += 1
        if patience >= args.early_stop:
            break

    end_time = time.time()
    # test all data inside dataset
    log_test = best_test_log
    acc_test = log_test['acc_test']
    bacc_test = log_test['bacc_test']
    roc_test = log_test['roc_test']
    macroF_test = log_test['macroF_test']
    if args.silhouette:
        silhouette_avg = ctrainer.calculate_silhouette(testloader, device, args.dataset, args.seed + count)
    formatted_log_info = {
        'acc_test': f"{acc_test:.4f}",
        'roc_test': f"{roc_test:.4f}",
        'macroF_test': f"{macroF_test:.4f}"
    }

    print(formatted_log_info)
    metric['acc'].append(acc_test)
    metric['macro_F'].append(macroF_test)
    metric['bacc'].append(bacc_test)
    metric['auroc'].append(roc_test)
    metric['pretrain_time'].append(mid_time - start_time)
    metric['train_time'].append(end_time - mid_time)
    metric['total_time'].append(end_time - start_time)
    metric['silhouette'].append(silhouette_avg)

acc = metric['acc']
macro_f1 = metric['macro_F']
b_acc = metric['bacc']
auc_roc = metric['auroc']
pretrain_time = metric['pretrain_time']
train_time = metric['train_time']
total_time = metric['total_time']
silhouette_score = metric['silhouette']
result = 'ACC: {:.4f}+{:.4f}, Macro-F: {:.4f}+{:.4f}, bACC: {:.4f}+{:.4}, ROC: {:.4f}+{:.4f}, silhouette_score: {:.4f}+{:.4f}'.format(
    np.mean(acc), np.std(acc), np.mean(macro_f1), np.std(macro_f1), np.mean(b_acc), np.std(b_acc), np.mean(auc_roc),
    np.std(auc_roc), np.mean(silhouette_score), np.std(silhouette_score))
print(acc)
print(silhouette_score)
print(result + '\n')
res_acc = '{:.2f}({:.2f})'.format(np.mean(acc) * 100, np.std(acc) * 100)
res_macro_f1 = '{:.2f}({:.2f})'.format(np.mean(macro_f1) * 100, np.std(macro_f1) * 100)
res_b_acc = '{:.2f}({:.2f})'.format(np.mean(b_acc) * 100, np.std(b_acc) * 100)
res_auc_roc = '{:.2f}({:.2f})'.format(np.mean(auc_roc) * 100, np.std(auc_roc) * 100)
res_pretrain_time = '{:.4f}'.format(np.mean(pretrain_time))
res_train_time = '{:.2f}'.format(np.mean(train_time))
res_total_time = '{:.2f}'.format(np.mean(total_time))
imbalance_ratio = '{:.2f}'.format(imbalance_ratio)
head_avg = '{:.2f}'.format(head_avg)
tail_avg = '{:.2f}'.format(tail_avg)
max_memory_allocated = torch.cuda.max_memory_allocated(device)
memory = '{:.2f}'.format(max_memory_allocated / (1024 ** 2))
header = ['split_mode', 'imb_ratio', 'head_avg', 'tail_avg', 'acc', 'b-acc', 'macro-f1', 'auc-roc',
          'train_time', 'memory', 'lr', 'wd', 'reweight_lr', 'reweight_weight']
new_experiment_result = [args.split_mode, imbalance_ratio, head_avg, tail_avg,
                         res_acc, res_b_acc, res_macro_f1, res_auc_roc, res_train_time, memory, args.lr,
                         args.weight_decay, args.reweight_lr, args.reweight_weight]
csv_file_path = "result/" + args.model + "/" + args.dataset + ".csv"
file_exists = os.path.exists(csv_file_path)
with open(csv_file_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    if not file_exists:
        writer.writerow(header)

    writer.writerow(new_experiment_result)
