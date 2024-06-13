from matplotlib.pyplot import plot
import torch
import numpy as np
import random
import os
import datasets as datasets
from torch_geometric.loader import DataLoader,RandomNodeSampler
import models.wl_models as wl_models
import models.models as models
import utils
import trainers.trainer as trainer
import math
import torch_geometric.datasets as tg_dataset
from torch_geometric.data import InMemoryDataset
import copy
from plots import plot_tsne,plot_dist1D,plot_chart
from tensorboardX import SummaryWriter
import ipdb
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

'''
###-------------------------------
initialization, load dataset
###------------------------------
'''

###configure arguments
args = utils.get_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device('cuda' if args.cuda else 'cpu')

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.log:
    tb_path = './tensorboard/{}/{}_wl/model{}/sup{}_lr{}layer{}topo{}_seed{}'.format(args.dataset,args.task, args.model,args.sup_ratio, args.lr,args.nlayer,args.topo_initial, args.seed)
    if not os.path.exists(tb_path):
        os.makedirs(tb_path)
    writer = SummaryWriter(tb_path)

###load dataset
if args.datatype == 'node':
    if args.dataset=='BA_shapes':
        dataset = datasets.SyncShapes('BA_shapes')
    elif args.dataset=='Tree_cycle':
        dataset = datasets.SyncShapes('Tree_cycle')
    elif args.dataset=='Tree_grid':
        dataset = datasets.SyncShapes('Tree_grid')
    elif args.dataset=='infected':
        dataset = datasets.infection()
    elif args.dataset=='LoadBA_shapes':
        dataset = datasets.LoadSyn('BA_shapes')
    elif args.dataset=='LoadTree_cycle':
        dataset = datasets.LoadSyn('Tree_cycle')
    elif args.dataset=='LoadTree_grid':
        dataset = datasets.LoadSyn('Tree_grid')
    elif args.dataset == 'cora':
        dataset = tg_dataset.Planetoid(root='./datasets/', name='Cora')
    elif args.dataset == 'cora_full':
        dataset = tg_dataset.CitationFull(root='./datasets/', name='Cora')
    elif args.dataset == 'chameleon' or args.dataset == 'squirrel':
        dataset = tg_dataset.WikipediaNetwork(root='./datasets/', name=args.dataset)
    else:
        ipdb.set_trace()
        print('error, unrecognized node classification dataset, {}'.format(args.dataset))


    args.nfeat = dataset[0].x.shape[-1]
    args.nclass = len(set(dataset[0].y.tolist()))
else:
    raise NotImplementedError("graph-type data is not implemented yet")

print(args)

'''
###------------------------------------------------
train MLP, get embeddings after PCA in numpy: trans_x
###------------------------------------------------
'''
if args.topo_initial=='mlp':
    MLPmodel = models.MLP(in_feat=args.nfeat, hidden_size=args.nhid, out_size=args.nclass, 
        layers=args.nlayer, dropout=args.dropout)
    MLPmodel = MLPmodel.to(device)
    MLP_TrainRatio = 0.7
    MLPtrainer = trainer.MLPTrainer(args, MLPmodel, dataset=dataset, batch_size=64, train_ratio=MLP_TrainRatio)

    ###train
    batch_num = math.ceil((MLPtrainer.train_idx.shape[0]/64))
    best_test_acc = 0
    for epoch in range(args.epochs):
        #test
        if epoch % 50 == 0:
            #datas is the revised graph for node classification
            log_test = {}
            if True:
                data = copy.deepcopy(dataset[0])
                log_info = MLPtrainer.test(data.to(device))
                for key in log_info:
                    if key not in log_test.keys():
                        log_test[key] = utils.meters(orders=1)
                        log_test[key].update(log_info[key], 1)                
            for key in log_info:
                if args.log:
                    writer.add_scalar('MLP'+key, log_test[key].avg(), epoch*batch_num) 
                #save embedding
                if log_info['acc_test'] > best_test_acc:
                    best_test_acc = log_info['acc_test']
                    MLP_emb = MLPmodel.embedding(data.x).detach().cpu().numpy()
                    np.save('Emb/{}.npy'.format(args.dataset),MLP_emb)
        #train
        if True:
            data = copy.deepcopy(dataset[0])
            for i in range(batch_num):
                log_info = MLPtrainer.train_batch(data.to(device), epoch, i)
                print('train log of MLP at epoch{} batch {}: {}'.format( epoch, i, log_info))

                for key in log_info:
                    if args.log:
                        writer.add_scalar('MLP'+key, log_info[key], epoch*batch_num+i)
elif args.topo_initial == 'emb':
    MLP_emb = dataset[0].x.cpu().numpy()
elif args.topo_initial == 'label':
    MLP_emb = dataset[0].y.cpu().numpy()


if args.topo_initial != 'label':
    x = MLP_emb
    pca = PCA(n_components=4)
    trans_x = pca.fit_transform(x)
    scale = 100/(trans_x.max()+0.000000000000001)
    trans_x = trans_x*scale
else:
    #for label, random sample acround center
    trans_x = np.stack([MLP_emb,np.zeros(MLP_emb.shape)], axis=1)
    noise_x = np.random.normal(0.0, 0.2, size=trans_x.shape[0])
    trans_x[:,0] = trans_x[:,0]+noise_x
    noise_y = np.random.normal(0.0, 0.5, size=trans_x.shape[0])
    trans_x[:,1] = trans_x[:,1]+noise_y


'''
###------------------------------------------------------------------------------
clustering, get cluter id for each node in numpy: clus, and revised data: datas
###------------------------------------------------------------------------------
'''

if args.topo_initial != 'label':
    eps=2
    min_sample= 4
    clust_model = DBSCAN(eps=eps, min_samples=min_sample).fit(trans_x)
    tried_iter = 0
    while len(set(clust_model.labels_)) <=3 or len(set(clust_model.labels_)) >=20 or (clust_model.labels_==-1).sum()>100:
        if len(set(clust_model.labels_)) <=3 and (clust_model.labels_== -1).sum()<100 :
            eps = eps*0.9
        else:
            eps = eps*1.1
        tried_iter +=1            
        clust_model = DBSCAN(eps=eps, min_samples=min_sample).fit(trans_x)
        if tried_iter >= 1000:
            print("cannot find suitable parameter for DBSCAN in obtaining anchors")
            raise Exception('fail to find DBSCAN paramter')
    clus = clust_model.labels_+2
    datas = dataset[0]
    datas.x = dataset[0].x.new(clus).reshape(-1).long()

    if args.log:
        writer.add_scalar('cluster number', len(set(clus)), 0)
        writer.add_scalar('cluster eps', eps, 0)
else:
    clus = dataset[0].y.cpu().numpy()
    datas = dataset[0]
    datas.x = dataset[0].x.new(clus).reshape(-1).long()



'''
###------------------------------------------------------------------------------
go through WL algorithm in torch.tensor: topo_labels, list of int: topo_size_list
###------------------------------------------------------------------------------
'''

wlmodel = wl_models.WL_model(args, nfeat=args.nfeat, 
                nhid=args.nhid, 
                nclass=args.nclass, 
                dropout=0,
                nlayer=args.nlayer)
wlmodel = wlmodel.to(device)
WLtrainer = trainer.ClsTrainer(args, wlmodel, dataset=dataset)

###train
steps = math.ceil((len(dataset)/args.batch_size))
best_test_acc = 0
for epoch in range(args.epochs):
    #test
    if epoch % 50 == 0:
        if True:
            #datas is the revised graph for node classification
            log_test = {}
            data = copy.deepcopy(datas)
            log_info = WLtrainer.test(data.to(device))
            for key in log_info:
                if key not in log_test.keys():
                    log_test[key] = utils.meters(orders=1)
                    log_test[key].update(log_info[key], 1)                
            for key in log_info:
                if args.log:
                    writer.add_scalar('wl_'+key, log_test[key].avg(), epoch*steps) 
    
    #train
    if True:
        data = copy.deepcopy(datas)
        log_info = WLtrainer.train_step(data.to(device), epoch)
        print('train log at  epoch{}: {}'.format( epoch, log_info))

        for key in log_info:
            if args.log:
                writer.add_scalar('wl_'+key, log_info[key], epoch*steps+1)


if args.log:
    writer.add_scalar('topology number', wlmodel.color_size, 0) 

    ###log visualization
    #draw embedding-class
    fig_class = plot_tsne(trans_x, data.y.cpu().numpy())
    writer.add_figure('emb_class',fig_class, 1)

    #draw embedding-cluster
    fig_clust = plot_tsne(trans_x, clus)
    writer.add_figure('emb_clust',fig_clust, 1)

    #get topo
    data = copy.deepcopy(datas)
    data = data.to(device)
    wlmodel = wlmodel.eval()
    topo_labels = wlmodel.wl(data.x, data.edge_index).detach()
    fig_topo = plot_tsne(trans_x, topo_labels.cpu().numpy())
    writer.add_figure('emb_topology_eval',fig_topo, 1)

topo_size_dict = {}
for topo in set(topo_labels.cpu().numpy()):
    topo_size = (topo_labels==topo).sum()
    topo_size_dict[topo] = topo_size.item()

'''
###-------------------------------
load GNN model: model
###------------------------------
'''

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
                nclass=args.nclass, 
                dropout=args.dropout,
                nlayer=args.nlayer)

if args.load is not None:
    model = utils.load_model(args,model,name='model_{}'.format(args.load))
model=model.to(device)

'''
###-------------------------------
train model
###------------------------------
'''

trainers=[]
Trainer_dict={'cls': trainer.ClsTrainer, 'gcls':trainer.GClsTrainer}

DOWNtrainer = Trainer_dict[args.task](args, model, dataset=dataset)
trainers.append(DOWNtrainer)

###train
steps = math.ceil((len(dataset)/args.batch_size))
best_test_acc = 0
for epoch in range(args.epochs):
    #test
    if epoch % 50 == 0:
        for ctrainer in trainers:
            #test all data inside dataset
            log_test = {}
            if True:
                data = copy.deepcopy(dataset[0])
                log_info = ctrainer.test(data.to(device))
                for key in log_info:
                    if key not in log_test.keys():
                        log_test[key] = utils.meters(orders=1)
                        log_test[key].update(log_info[key], 1)                
            for key in log_info:
                if args.log:
                    writer.add_scalar(key, log_test[key].avg(), epoch*steps) 

    for trainer in trainers:#not train if model is loaded
        #train
        if args.load is None:
            data = copy.deepcopy(dataset[0])
            log_info = trainer.train_step(data.to(device), epoch)
            print('train log of {} in epoch{}: {}'.format(args.model, epoch, log_info))

            for key in log_info:
                if args.log:
                    writer.add_scalar(key, log_info[key], epoch*steps+1)


'''
###-------------------------------
test performance
###------------------------------
'''

###compute accuracy in topology-wise manner
for ctrainer in trainers:
    data = copy.deepcopy(dataset[0])
    topo_ac_dict = ctrainer.test_all_topo(data.to(device), topo_labels)
    
##check correspondence between topology and class
topo_consis_dict, micro_consis_score, macro_consis_score = utils.group_consistency(topo_labels.cpu().numpy(), data.y.cpu().numpy())
if args.log:
    writer.add_scalar('micro_topo_consistency', micro_consis_score, 1)
    writer.add_scalar('macro_topo_consistency', macro_consis_score, 1)

##draw figures
topo_ac_list=[]
topo_cons_list=[]
topo_size_list=[]
for topo in set(topo_labels.cpu().numpy()):
    topo_ac_list.append(topo_ac_dict[topo])
    topo_cons_list.append(topo_consis_dict[topo])
    topo_size_list.append(topo_size_dict[topo])

sensible_topo_idx = np.array(topo_size_list)>1

fig = plot_chart([np.array(topo_size_list)], name_list=['size of each topology group'], x_start=1,x_name='topo group',y_name='size')
if args.log:
    writer.add_figure('topo_group size',fig, 1)
fig = plot_chart([np.array(topo_ac_list)], name_list=['acc on each topo group'], x_start=1,x_name='topo group')
if args.log:
    writer.add_figure('topo_group accuracy',fig, 1)
fig = plot_chart([np.array(topo_cons_list)], name_list=['consistency on each topo group'], x_start=1,x_name='topo group')
if args.log:
    writer.add_figure('topo_group consistency',fig, 1)

fig = plot_chart([np.array(topo_ac_list)[sensible_topo_idx]], name_list=['acc on each topo group'], x_start=1,x_name='topo group')
if args.log:
    writer.add_figure('sensible topo_group accuracy',fig, 1)
fig = plot_chart([np.array(topo_cons_list)[sensible_topo_idx]], name_list=['consistency on each topo group'], x_start=1,x_name='topo group')
if args.log:
    writer.add_figure('sensible topo_group consistency',fig, 1)    
fig = plot_dist1D(np.array(topo_cons_list)[sensible_topo_idx], label=np.array(topo_size_list)[sensible_topo_idx])#
if args.log:
    writer.add_figure('sensible distribution of topo_wise consistency',fig, 1)
fig = plot_dist1D(np.array(topo_ac_list)[sensible_topo_idx], label=np.array(topo_size_list)[sensible_topo_idx])#
if args.log:
    writer.add_figure('sensible distribution of topo_wise accuracy',fig, 1)

#draw only test performance
data = copy.deepcopy(dataset[0])
test_topo_ac_dict = ctrainer.test_all_topo(data.to(device), topo_labels, only_test=True)
test_topo_ac_list=[]
test_topo_size_list=[]
for topo in test_topo_ac_dict.keys():
    test_topo_ac_list.append(test_topo_ac_dict[topo])
    test_topo_size_list.append(topo_size_dict[topo])
test_sensible_topo_idx = np.array(test_topo_size_list)>1
fig = plot_dist1D(np.array(test_topo_ac_list)[test_sensible_topo_idx], label=np.array(test_topo_size_list)[test_sensible_topo_idx])#
if args.log:
    writer.add_figure('sensible distribution of test-time topo_wise accuracy',fig, 1)

'''
###-------------------------------
exit
###------------------------------
'''

###save configuration, model parameters
if args.save:
    utils.save_args(args)

if args.log:
    writer.close()

print("Optimization Finished!")