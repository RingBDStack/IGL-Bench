from sklearn.preprocessing import label_binarize
import torch
import os
import argparse
import json

from sklearn.metrics import roc_auc_score, f1_score
import torch.nn.functional as F
import ipdb
import numpy as np
from sklearn.cluster import SpectralClustering, DBSCAN
import random
from scipy.spatial.distance import pdist,squareform
import yaml


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imb_ratio', type=float, default=0.9)
    parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
    parser.add_argument('--sparse', action='store_true', default=False,
                    help='whether use sparse adj matrix')
    parser.add_argument('--seed', type=int, default=10)
    
    parser.add_argument('--datatype', type=str, default='graph', choices=['node', 'graph'])
    parser.add_argument('--task', type=str, default='cls', choices=['cls', 'gcls', 'expl', 'reweight'])#cls: node classification; gcls: graph classification; expl: explanation
    parser.add_argument('--dataset', type=str, default='BA_shapes') #choices=['BA_shapes','infected','Tree_cycle','Tree_grid','LoadBA_shapes', 'LoadTree_cycle','LoadTree_grid','mutag', 'SpuriousMotif_{}'.format(mix_ratio), 'SST2','SST5','Twitter']
    
    parser.add_argument('--nlayer', type=int, default=2)#intermediate feature dimension
    parser.add_argument('--nhid', type=int, default=128)#intermediate feature dimension
    parser.add_argument('--nclass', type=int, default=2)#number of labels
    parser.add_argument('--nfeat', type=int, default=64) # input feature dimension
    parser.add_argument('--epochs', type=int, default=500,
                help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=128,
                help='Number of batches inside an epoch.')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_nums', type=int, default=6000, help='number of batches per epoch')

    parser.add_argument('--sup_ratio', type=float, default=0.1)
    parser.add_argument('--val_ratio', type=float, default=0.3)
    parser.add_argument('--test_ratio', type=float, default=0.6)
    parser.add_argument('--res', type=bool, default=False)

    parser.add_argument('--load', type=int, default=None) #load from pretrained model under the same setting, indicate load epoch
    parser.add_argument('--save', action='store_true', default=False,help='whether save checkpoints')
    parser.add_argument('--log', action='store_true', default=False,
                    help='whether creat tensorboard logs')
    parser.add_argument('--load_model', type=str, default=None) #To indicate pre-train in other folders. Like "./checkpoint/SpuriousMotif_0.3/best".
    

    parser.add_argument('--model', type=str, default='gcn', 
        choices=['sage','gcn', 'gin', 'wl_model'])
    parser.add_argument('--shared_encoder', action='store_true', default=False,help='False: train one end-to-end model; True: for multi-task, train a shared encoder with task-wise heads')
    
    parser.add_argument('--load_config', action='store_true', default=False, help='whether load training configurations')

    #explainer choices
    parser.add_argument('--explainer', type=str, default='gnnexplainer', choices=['gnnexplainer', 'pgexplainer','pgexplainer2' ])
    parser.add_argument('--directional', action='store_true', default=False, help='whether taking graph as directional or not in explanation')
    parser.add_argument('--edge_size', type=float, default=0.05, help='control edge mask sparsity')
    parser.add_argument('--edge_ent', type=float, default=1.0, help='control edge entropy weight')
    parser.add_argument('--expl_loss', type=str, default='Tgt', choices=['Tgt', 'Entropy','Dif' ])#
    parser.add_argument('--aligner', type=str, default='emb', choices=['emb', 'anchor','both'])#
    parser.add_argument('--aligner_combine_weight', type=float, default=1.0)#
    parser.add_argument('--align_emb', action='store_true', default=False,  help='whether aligning embeddings in obtaining explanation')
    parser.add_argument('--align_with_grad', action='store_true', default=False,  help='whether aligning embeddings in obtaining explanation with gradient-based weighting')
    parser.add_argument('--align_weight', type=float, default=1.0)
    parser.add_argument('--split', type=int, default=0) # 0: split_graph(); 1: split_graph_arti()

    #topo-test choices
    parser.add_argument('--topo_initial', type=str, default='label', choices=['label', 'mlp','emb' ])
    parser.add_argument('--reweight_lr', type=float, default=0.01)
    parser.add_argument('--reweight_weight', type=float, default=0.2)
    parser.add_argument('--reweighter', type=str, default='struct', choices=['struct','structATT', 'class', 'gcn'])
    parser.add_argument('--n_mem', type=int, nargs='+', default=[8,8,8,8,8], help='List of memory sizes for each layer')
    parser.add_argument('--use_key', action='store_true', default=False, help='whether use key in memory')
    parser.add_argument('--att', type=str, default='dp', choices=['dp','mlp'])
    parser.add_argument('--adv_step', type=int, default=1)
    parser.add_argument('--EM', action='store_true', default=False, help='whether use EM update')
    parser.add_argument('--pretrain_reweighter', action='store_true', default=False, help='whether pretrain the reweighter')

    #setting for ImbTopo dataset
    parser.add_argument('--intra_im_ratio', type=float, default=0.1)
    parser.add_argument('--inter_im_ratio', type=float, default=0.6)
    parser.add_argument('--reweight_task', type=str, nargs='+') #choices=['cls', 'wlcls']

    #baseline
    parser.add_argument('--baseline', type=str, default='no', choices=['no','reweight', 'EMsmote', 'focal', 'oversample'])
    

    return parser

def get_args():
    parser = get_parser()
    args = parser.parse_args()

    if args.load_config:
        config_path='./configs/{}/{}/{}'.format(args.task,args.dataset,args.model)
        with open(config_path) as f:
            args.__dict__ = json.load(f)

    return args

def save_args_to_yaml(args, config_folder='config'):
    # 构建数据集特定的文件夹路径
    dataset_folder = os.path.join(config_folder, args.dataset)
    # 如果数据集文件夹不存在，创建它
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    # 构建具体的文件路径
    filename = os.path.join(dataset_folder, f"{args.model}.yml")

    # 将命令行参数转换为字典，并保存为YAML文件
    with open(filename, 'w') as file:
        yaml.dump(args.__dict__, file, default_flow_style=False)
        
def save_args(args):
    config_path='./configs/{}/{}/'.format(args.task,args.dataset)

    if not os.path.exists(config_path):
        os.makedirs(config_path)

    with open(config_path+args.model,'w+') as f:
        json.dump(args.__dict__, f, indent=2)

    return

def clust(features, n_clusters=8):
    clustering = SpectralClustering(n_clusters=n_clusters, assign_labels='discretize', random_state=0, affinity='nearest_neighbors').fit(features)

    return clustering.labels_


def save_model(args, model, epoch=None, name='model'):
    saved_content = {}
    saved_content[name] = model.state_dict()

    path = './checkpoint/{}/{}'.format(args.dataset, args.model)
    if not os.path.exists(path):
        os.makedirs(path)

    #torch.save(saved_content, 'checkpoint/{}/{}_epoch{}_edge{}_{}.pth'.format(args.dataset,args.model,epoch, args.used_edge, args.method))
    if epoch is not None:
        torch.save(saved_content, os.path.join(path,'{}_{}.pth'.format(name, epoch)))
        print("successfully saved: {}".format(epoch))
    else:
        torch.save(saved_content, os.path.join(path,'{}.pth'.format(name)))
        print("successfully saved: {}".format(name))

    return

def load_model(args, model, name='model_500'):
    
    loaded_content = torch.load('./checkpoint/{}/{}/{}.pth'.format(args.dataset, args.model,name), map_location=lambda storage, loc: storage)

    model.load_state_dict(loaded_content['best'])

    print("successfully loaded: {}.pth".format(name))

    return model

def load_specific_model(model, name='./checkpoint/mutag/gcn/model_500.pth'):
    
    loaded_content = torch.load(name, map_location=lambda storage, loc: storage)
    model.load_state_dict(loaded_content['best'])

    print("successfully loaded: {}".format(name))

    return model

def path2mask(head_node, path, edge_index):
    #change path to edge mask
    edge_mask = edge_index.new(edge_index.shape[1]).fill_(0)
    
    cur_ind = 0
    while path[cur_ind+1] != -1:
        src_node = path[cur_ind+1]
        tgt_node = path[cur_ind]

        edge_ind = torch.mul(edge_index[0]==src_node, edge_index[1]==tgt_node).nonzero().squeeze().item()
        edge_mask[edge_ind] = 1
        cur_ind += 1

    return edge_mask.to(int)

def accuracy(logits, labels):
    preds = logits.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def grouped_accuracy(logits, labels, group_labels):
    # all inputs should be stored in numpy array

    preds = logits.argmax(1)
    group_ac_dict={}
    for group in set(group_labels):
        group_idx = group_labels==group
        group_ac = (preds[group_idx]==labels[group_idx]).sum()/(group_idx.sum()+0.00000001)
        group_ac_dict[group] = group_ac

    return group_ac_dict

def grouped_measure(measures, group_labels):
    # all inputs should be stored in numpy array

    group_measure_dict={}
    for group in set(group_labels):
        group_idx = group_labels==group
        group_measure = (measures[group_idx]).sum(0)/(group_idx.sum()+0.00000001)
        group_measure_dict[group] = group_measure

    return group_measure_dict

def group_consistency(group_labels, labels):

    micro_consis = meters()
    macro_consis = meters()

    group_cons_dict={}
    for group in set(group_labels):
        group_idx = group_labels==group
        label_count = np.bincount(labels[group_idx])
        group_cons_dict[group] = label_count.max()/group_idx.sum()
        
        micro_consis.update(group_cons_dict[group],group_idx.sum())
        macro_consis.update(group_cons_dict[group],1)

    return group_cons_dict, micro_consis.avg(), macro_consis.avg()


def print_class_acc(logits, labels, pre='valid'):
    pre_num = 0
    #print class-wise performance
    
    for i in range(labels.max()+1):
        index_pos = labels==i
        cur_tpr = accuracy(logits[index_pos], labels[index_pos])
        print(str(pre)+" class {:d} True Positive Rate: {:.3f}".format(i,cur_tpr.item()))

        index_neg = labels != i
        labels_neg = labels.new(labels.shape).fill_(i)
        
        cur_fpr = accuracy(logits[index_neg,:], labels_neg[index_neg])
        print(str(pre)+" class {:d} False Positive Rate: {:.3f}".format(i,cur_fpr.item()))
    

    if labels.max() > 1:
        auc_score = roc_auc_score(labels.detach().cpu(), F.softmax(logits, dim=-1).detach().cpu(), average='macro', multi_class='ovr')
    else:
        auc_score = roc_auc_score(labels.detach().cpu(), F.softmax(logits, dim=-1)[:,1].detach().cpu(), average='macro')

    macro_F = f1_score(labels.detach().cpu(), torch.argmax(logits, dim=-1).detach().cpu(), average='macro')
    print(str(pre)+' current auc-roc score: {:f}, current macro_F score: {:f}'.format(auc_score,macro_F))

    return

def Roc_F(logits, labels, pre='valid'):
    pre_num = 0
    #print class-wise performance
    '''
    for i in range(labels.max()+1):
        
        cur_tpr = accuracy(logits[pre_num:pre_num+class_num_list[i]], labels[pre_num:pre_num+class_num_list[i]])
        print(str(pre)+" class {:d} True Positive Rate: {:.3f}".format(i,cur_tpr.item()))

        index_negative = labels != i
        labels_negative = labels.new(labels.shape).fill_(i)
        
        cur_fpr = accuracy(logits[index_negative,:], labels_negative[index_negative])
        print(str(pre)+" class {:d} False Positive Rate: {:.3f}".format(i,cur_fpr.item()))

        pre_num = pre_num + class_num_list[i]
    '''

    if labels.max() > 1:#require set(labels) to be the same as columns of logits 
        auc_score = roc_auc_score(labels.detach().cpu(), F.softmax(logits, dim=-1).detach().cpu(), average='macro', multi_class='ovr')
    else:
        auc_score = roc_auc_score(labels.detach().cpu(), F.softmax(logits, dim=-1)[:,1].detach().cpu(), average='macro')

    macro_F = f1_score(labels.detach().cpu(), torch.argmax(logits, dim=-1).detach().cpu(), average='macro')
    #print(str(pre)+' current auc-roc score: {:f}, current macro_F score: {:f}'.format(auc_score,macro_F))

    return auc_score, macro_F

def groupewise_perform(logits, labels, group_label, pre='valid'):
    pre_num = 0
    #print class-wise performance
    
    accs = meters()
    aucs = meters()
    Fs = meters()

    for i in range(group_label.max()+1):
        index_pos = group_label==i
        cur_acc = accuracy(logits[index_pos], labels[index_pos]).item()
        #cur_auc = roc_auc_score(labels[index_pos].detach().cpu(), F.softmax(logits[index_pos], dim=-1).detach().cpu(), average='macro', multi_class='ovr')
        cur_auc = 0 #not computed for now
        cur_F = f1_score(labels[index_pos].detach().cpu(), torch.argmax(logits[index_pos], dim=-1).detach().cpu(), average='macro')
        
        accs.update(cur_acc)
        aucs.update(cur_auc)
        Fs.update(cur_F)

    return accs.avg(), aucs.avg(), Fs.avg()

class meters:
    '''
    collects the results at each inference batch, and return the result in total
    param orders: the order in updating values
    '''
    def __init__(self, orders=1):
        self.avg_value = 0
        self.tot_weight = 0
        self.orders = orders
        
    def update(self, value, weight=1.0):
        value = float(value)

        if self.orders == 1:
            update_step = self.tot_weight/(self.tot_weight+weight)
            self.avg_value = self.avg_value*update_step + value*(1-update_step)
            self.tot_weight += weight
        

    def avg(self):

        return self.avg_value

def class_weight(ini_dataset):
    class_weights = []
    tot_num = ini_dataset[0].y.shape[0]
    class_num = len(set(ini_dataset[0].y.cpu().numpy()))
    for y in range(class_num):
        class_size = (ini_dataset[0].y==y).sum().item()
        assert class_size !=0, "error in computing class_weight, size of class {} is 0".format(y)
        class_weight = tot_num/(class_num*class_size+0.000001)
        class_weights.append(class_weight)

    return class_weights

def split_dataset(dataset, sup_ratio, im_ratio=1, art_balance=True, train_num=0):
    # art_balance: whether manipulate balance status of each class. if set to false,  im_ratio will not be used

    if art_balance:
        label_list=[]

        data_num = len(dataset)
        for i in range(data_num):
            label_list.append(dataset[i].y.item())
        class_num = len(set(label_list))
        
        label_array = np.array(label_list)
    
        if train_num == 0:
            train_num = int(data_num/class_num*sup_ratio) #expected train number for each class

        train_idx = []
        val_idx = []
        test_idx = []
        for i, c in enumerate(set(label_list)):
            c_idx = (label_array==c).nonzero()[0]

            if c < class_num/2:
                train_idx = train_idx + c_idx[:train_num].tolist()
                val_idx = val_idx + c_idx[train_num:train_num+int((len(c_idx)-train_num)/3)].tolist()
                test_idx = test_idx + c_idx[train_num+int((len(c_idx)-train_num)/3):].tolist()
            else:
                c_train_num = max(int(train_num*im_ratio),1)
                train_idx = train_idx + c_idx[:c_train_num].tolist()
                val_idx = val_idx + c_idx[c_train_num:c_train_num+int((len(c_idx)-c_train_num)/3)].tolist()
                test_idx = test_idx + c_idx[c_train_num+int((len(c_idx)-c_train_num)/3):].tolist()

        train_mask = dataset[0].y.new(label_array.shape[0]).fill_(0).bool()
        train_mask[train_idx] = 1
        val_mask = dataset[0].y.new(label_array.shape[0]).fill_(0).bool()
        val_mask[val_idx] = 1
        test_mask = dataset[0].y.new(label_array.shape[0]).fill_(0).bool()
        test_mask[test_idx] = 1
    else:
        data_num = len(dataset)
        train_mask = dataset[0].y.new(data_num).fill_(0).bool()
        train_mask[:int(data_num*sup_ratio)] = 1
        val_mask = dataset[0].y.new(data_num).fill_(0).bool()
        val_mask[int(data_num*sup_ratio):int(data_num*(sup_ratio+0.1))] = 1
        test_mask = dataset[0].y.new(data_num).fill_(0).bool()
        test_mask[int(data_num*(sup_ratio+0.1)):] = 1


    return train_mask, val_mask, test_mask



def emb_smote(embed, labels, idx_train=None, portion=1.0, im_class_num=3):
    # over-sample the embedding of last several classes
    # args: 
    #    embed, labels: torch.tensor()
    #    portion: 0 for auto-assign over-sample portion; (0,1] for given portion

    if idx_train is None:
        idx_train = labels.new(embed.shape[0]).fill_(1)
    c_largest = labels.max().item()
    avg_number = int(idx_train.shape[0]/(c_largest+1))
    #ipdb.set_trace()

    for i in range(im_class_num):
        chosen = idx_train[(labels==(c_largest-i))[idx_train]]
        num = int(chosen.shape[0]*portion)
        if portion == 0:
            c_portion = int(avg_number/chosen.shape[0])
            num = chosen.shape[0]
        else:
            c_portion = 1

        for j in range(c_portion):
            chosen = chosen[:num]
            if num >0:

                chosen_embed = embed[chosen,:]
                distance = squareform(pdist(chosen_embed.cpu().detach()))
                np.fill_diagonal(distance,distance.max()+100)

                idx_neighbor = distance.argmin(axis=-1)
                
                interp_place = random.random()
                new_embed = embed[chosen,:] + (chosen_embed[idx_neighbor,:]-embed[chosen,:])*interp_place


                new_labels = labels.new(torch.Size((chosen.shape[0],1))).reshape(-1).fill_(c_largest-i)
                idx_new = np.arange(embed.shape[0], embed.shape[0]+chosen.shape[0])
                idx_train_append = idx_train.new(idx_new)

                embed = torch.cat((embed,new_embed), 0)
                labels = torch.cat((labels,new_labels), 0)
                idx_train = torch.cat((idx_train,idx_train_append), 0)

    
    return embed, labels, idx_train

def graph_statistics(data, node_idx=None, ):


    return