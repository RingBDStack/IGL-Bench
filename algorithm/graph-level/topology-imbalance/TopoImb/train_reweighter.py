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
from trainers.EMTrainer import EMClsTrainer, EMGClsTrainer
import math
import torch_geometric.datasets as tg_dataset
from torch_geometric.data import InMemoryDataset
import copy
from plots import plot_tsne,plot_dist1D,plot_chart, plot_single_grid
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
    tb_path = './tensorboard/{}/{}_trainReweighter/model{}/sup{}lr{}layer{}topo{}batch{}dropout{}seed{}/reweight{}lr{}nmem{}att{}usekey{}advstep{}EM{}'.format(args.dataset,args.task, args.model,args.sup_ratio, args.lr,args.nlayer,args.topo_initial, args.batch_size,args.dropout,args.seed,
        args.reweight_weight,args.reweight_lr, args.n_mem, args.att, args.use_key, args.adv_step, args.EM)
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
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    mem_dataloader = DataLoader(dataset, batch_size=args.batch_size*2, shuffle=False)
    testloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

elif args.datatype == 'graph':
    if args.dataset == 'mutag':
        dataset = datasets.MoleculeDataset(root='./datasets/', name='MUTAG')
        dataset = dataset.shuffle()
        args.nclass = 2
    elif args.dataset.split('_')[0] == 'SpuriousMotif':
        name, mix_ratio = args.dataset.split('_')
        mix_ratio = float(mix_ratio)
        dataset = datasets.SyncGraphs(name, mix_ratio=mix_ratio)
        dataset = dataset.shuffle()
        args.nclass = 3
    elif args.dataset == 'ImbTopo':
        dataset = datasets.ImGraphs('ImbTopo',intra_im_ratio=args.intra_im_ratio, inter_im_ratio=args.inter_im_ratio)
        dataset = dataset.shuffle()
        args.nclass = 3

    args.nfeat = dataset[0].x.shape[-1]
    graph_num = len(dataset)
    
    dataloader = DataLoader(dataset[:int(graph_num*args.sup_ratio)], batch_size=args.batch_size, shuffle=True)
    mem_dataloader = DataLoader(dataset[:int(graph_num*args.sup_ratio)], batch_size=args.batch_size*2, shuffle=True)
    valoader = DataLoader(dataset[int(graph_num*args.sup_ratio):int(graph_num*(args.sup_ratio+args.val_ratio))], batch_size=args.batch_size, shuffle=False)
    testloader = DataLoader(dataset[-int(graph_num*args.test_ratio):], batch_size=len(dataset[int(graph_num*args.test_ratio):]), shuffle=False)

    allloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    #raise NotImplementedError("graph-type data is not implemented yet")

print(args)


'''
###-------------------------------
initialize/load GNN model
###------------------------------
'''

if args.datatype == 'node':
    
    reweighter = models.StructGNN(args,nfeat=args.nfeat, 
                nhid=args.nhid, 
                nclass=args.nclass, 
                dropout=0,
                n_mem=args.n_mem,
                nlayer=2,
                use_key=args.use_key,
                att=args.att)
    reweighter = reweighter.to(device)

elif args.datatype == 'graph':
    reweighter = models.StructGraphGNN(args,nfeat=args.nfeat, 
                nhid=args.nhid, 
                nclass=args.nclass, 
                dropout=0,
                n_mem=args.n_mem,
                nlayer=2,
                use_key=args.use_key,
                att=args.att)


if args.load is not None:
    reweighter = utils.load_model(args,reweighter,name='reweighter_{}'.format(args.load))

reweighter = reweighter.to(device)



'''
###-------------------------------
train: model
###------------------------------
'''
re_trainers=[] #auxiliary tasks for reweighter
if args.datatype == 'node':
    retrainer = EMClsTrainer(args,reweighter,dataset=dataset)
elif args.datatype == 'graph':
    retrainer = EMGClsTrainer(args, reweighter, dataset=dataset)

re_trainers.append(retrainer)

###pretrain reweighter
#ipdb.set_trace()
#m = copy.deepcopy(reweighter)
steps=math.ceil((len(dataset)/args.batch_size))
for epoch in range(args.epochs):
    if epoch % 50 == 0:
        for re_trainer in re_trainers:
            #test all data inside dataset
            log_test = {}
            for batch, data in enumerate(testloader):
                log_info = re_trainer.test(data.to(device))
                for key in log_info:
                    if key not in log_test.keys():
                        log_test[key] = utils.meters(orders=1)
                    log_test[key].update(log_info[key], 1)                
            for key in log_info:
                if args.log:
                    writer.add_scalar('reweighter_train_'+key, log_test[key].avg(), epoch*steps) 

            # ---------------------
            # topo-wise performance 
            # ---------------------  
            if args.datatype=='graph':
                for batch, data in enumerate(allloader):
                    all_data = data
            elif args.datatype=='node':
                all_data = dataset[0]

            if 'topo_label' in all_data.keys:
                topo_labels = all_data.topo_label.to(device)
            else:
                topo_labels = all_data.y.to(device)

            #draw memory selection
            topo_ac_dict, topo_sim_dict_list = re_trainer.test_all_topo(all_data.to(device), topo_labels, return_sims=True)
            topo_ac_list=[]
            topo_sim_list_list = [ [] for diction in topo_sim_dict_list]
            for topo in set(topo_labels.cpu().numpy()):
                topo_ac_list.append(topo_ac_dict[topo])
                for i, diction in enumerate(topo_sim_dict_list):
                    topo_sim_list_list[i].append(diction[topo])

            fig = plot_chart([np.array(topo_ac_list)], name_list=['accuray of each topology group'], x_start=1,x_name='topo group',y_name='acc')
            if args.log:
                writer.add_figure('topo_group ac',fig, epoch*steps)
            for i,sim_list in enumerate(topo_sim_list_list):
                fig = plot_single_grid(np.array(sim_list), name='memory selection at layer {}'.format(i), x_start=1,x_name='topo group',y_name='memory selection dist')
                if args.log:
                    writer.add_figure('topo_group memory selection at layer {}'.format(i), fig, epoch*steps)

            #draw T-SNE
            emb_list = retrainer.get_emb(all_data.to(device))
            for i,emb in enumerate(emb_list):
                fig = plot_tsne(emb.cpu().numpy(), topo_labels.cpu().numpy())
                if args.log:
                    writer.add_figure('TSNE at layer {}'.format(i), fig, epoch*steps)
            #save model
            if args.save and epoch%50==0:
                utils.save_model(args, reweighter,epoch=epoch, name='reweighter')
                if log_info['acc_test'] > best_test_acc:
                    best_test_acc = log_info['acc_test']
                    utils.save_model(args,reweighter,name='best')

    for re_trainer in re_trainers:
        if args.EM:
            for batch,data in enumerate(dataloader):
                log_info = re_trainer.train_step(data.to(device), epoch)
                print('reweighter train log at {} in epoch{}: {}'.format(batch, epoch, log_info))
                for key in log_info:
                    if args.log:
                        writer.add_scalar('reweighter_train_'+key, log_info[key], epoch*steps+batch+1)
            for batch, data in enumerate(mem_dataloader):
                log_info = re_trainer.train_mem_step(data.to(device), epoch)
                print('reweighter mem train log at {} in epoch{}: {}'.format(batch, epoch, log_info))
                for key in log_info:
                    if args.log:
                        writer.add_scalar('reweighter_train_'+key, log_info[key], epoch*steps+batch+1)
        else:
            for batch,data in enumerate(dataloader):
                log_info = re_trainer.train_all_step(data.to(device), epoch)
                print('reweighter train log at {} in epoch{}: {}'.format(batch, epoch, log_info))
                for key in log_info:
                    if args.log:
                        writer.add_scalar('reweighter_train_'+key, log_info[key], epoch*steps+batch+1)


