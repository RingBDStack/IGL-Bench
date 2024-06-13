import random
import codecs
import copy
import math
import os,sys

import torch
import torch.nn.functional as F
import numpy as np
from utils import index2dense
from scipy.sparse import csr_matrix
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Amazon,Actor,WikipediaNetwork
from ogb.nodeproppred import PygNodePropPredDataset

#access a quantity-balanced training set: each class has the same training size train_each
def get_split(opt,all_idx,all_label,nclass):
    train_each = (len(all_idx)*0.1) // nclass 
    valid_each = (len(all_idx)*0.1) // nclass 
    if opt.data_name == 'arxiv':
        all_label = np.squeeze(all_label)
    train_list = [0 for _ in range(nclass)]
    train_node = [[] for _ in range(nclass)]
    train_idx  = []
    
    for iter1 in all_idx:
        iter_label = all_label[iter1]
        if train_list[iter_label] < train_each:
            train_list[iter_label]+=1
            train_node[iter_label].append(iter1)
            train_idx.append(iter1)

        if sum(train_list)==train_each*nclass:break
    assert sum(train_list)==train_each*nclass
    after_train_idx = list(set(all_idx)-set(train_idx))
    #random.shuffle(after_train_idx)

    valid_list = [0 for _ in range(nclass)]
    valid_idx  = []
    for iter2 in after_train_idx:
        iter_label = all_label[iter2]
        if valid_list[iter_label] < valid_each:
            valid_list[iter_label]+=1
            valid_idx.append(iter2)
        if sum(valid_list)==valid_each*nclass:break

    assert sum(valid_list)==valid_each*nclass
    test_idx = list(set(after_train_idx)-set(valid_idx))

    return train_idx,valid_idx,test_idx,train_node
def get_split_arxiv(opt, all_idx, all_label, nclass):
    num_samples = len(all_idx)
    train_each = int(num_samples * 0.1 / nclass)  
    valid_each = int(num_samples * 0.1 / nclass)  

    all_label = np.squeeze(all_label)
    label_counts = np.bincount(all_label, minlength=nclass) 

    train_list = [0] * nclass
    train_node = [[] for _ in range(nclass)]
    train_idx = []
    
    valid_list = [0] * nclass
    valid_idx = []

    after_train_idx = []

    train_class = [min(train_each, int(count * 0.1)) for count in label_counts]
    valid_class = [min(valid_each, int(count * 0.1)) for count in label_counts]

    for idx in all_idx:
        label = all_label[idx]
        if train_list[label] < train_class[label]:
            train_list[label] += 1
            train_node[label].append(idx)
            train_idx.append(idx)
        else:
            after_train_idx.append(idx)

    for idx in after_train_idx:
        label = all_label[idx]
        if valid_list[label] < valid_class[label]:
            valid_list[label] += 1
            valid_idx.append(idx)
    
    test_idx = list(set(after_train_idx) - set(valid_idx))
    
    return train_idx, valid_idx, test_idx, train_node
#access a quantity-imbalanced training set; the training set follows the step distribution.
def get_step_split(opt,all_idx,all_label,nclass=7):
    if opt.data_name == 'arxiv':
        all_label = np.squeeze(all_label)
    base_valid_each = len(all_idx)*0.1

    imb_ratio = opt.imb_ratio
    head_list = opt.head_list if len(opt.head_list)>0 else [i for i in range(nclass//2)]

    all_class_list = [i for i in range(nclass)]
    tail_list = list(set(all_class_list) - set(head_list))

    h_num = len(head_list)
    t_num = len(tail_list)

    #base_train_each = int( len(all_idx) * opt.labeling_ratio / (t_num + h_num * imb_ratio) )
    base_train_each = len(all_idx)*0.1

    idx2train,idx2valid = {},{}

    total_train_size = 0
    total_valid_size = 0

    for i_h in head_list: 
        idx2train[i_h] = int(base_train_each * 1)
        idx2valid[i_h] = int(base_valid_each * 1) 

        total_train_size += idx2train[i_h]
        total_valid_size += idx2valid[i_h]

    for i_t in tail_list: 
        idx2train[i_t] = int(base_train_each * 1)
        idx2valid[i_t] = int(base_valid_each * 1)

        total_train_size += idx2train[i_t]
        total_valid_size += idx2valid[i_t]

    train_list = [0 for _ in range(nclass)]
    train_node = [[] for _ in range(nclass)]
    train_idx  = []

    for iter1 in all_idx:
        iter_label = all_label[iter1]
        if train_list[iter_label] < idx2train[iter_label]:
            train_list[iter_label]+=1
            train_node[iter_label].append(iter1)
            train_idx.append(iter1)

        if sum(train_list)==total_train_size:break

    assert sum(train_list)==total_train_size

    after_train_idx = list(set(all_idx)-set(train_idx))

    valid_list = [0 for _ in range(nclass)]
    valid_idx  = []
    for iter2 in after_train_idx:
        iter_label = all_label[iter2]
        if valid_list[iter_label] < idx2valid[iter_label]:
            valid_list[iter_label]+=1
            valid_idx.append(iter2)
        if sum(valid_list)==total_valid_size:break

    assert sum(valid_list)==total_valid_size
    test_idx = list(set(after_train_idx)-set(valid_idx))

    return train_idx,valid_idx,test_idx,train_node


#return the ReNode Weight
def get_renode_weight(opt,data):

    ppr_matrix = data.Pi  #personlized pagerank
    gpr_matrix = torch.tensor(data.gpr).float() #class-accumulated personlized pagerank

    base_w  = opt.rn_base_weight
    scale_w = opt.rn_scale_weight
    nnode = ppr_matrix.size(0)
    unlabel_mask = data.train_mask.int().ne(1)#unlabled node


    #computing the Totoro values for labeled nodes
    gpr_sum = torch.sum(gpr_matrix,dim=1)
    gpr_rn  = gpr_sum.unsqueeze(1) - gpr_matrix
    rn_matrix = torch.mm(ppr_matrix,gpr_rn)

    labels = data.y.squeeze() 
    label_matrix = F.one_hot(labels, gpr_matrix.size(1)).float()
    label_matrix[unlabel_mask] = 0

    rn_matrix = torch.sum(rn_matrix * label_matrix,dim=1)
    rn_matrix[unlabel_mask] = rn_matrix.max() + 99 #exclude the influence of unlabeled node
    
    #computing the ReNode Weight
    train_size    = torch.sum(data.train_mask.int()).item()
    totoro_list   = rn_matrix.tolist()
    id2totoro     = {i:totoro_list[i] for i in range(len(totoro_list))}
    sorted_totoro = sorted(id2totoro.items(),key=lambda x:x[1],reverse=False)
    id2rank       = {sorted_totoro[i][0]:i for i in range(nnode)}
    totoro_rank   = [id2rank[i] for i in range(nnode)]
    
    rn_weight = [(base_w + 0.5 * scale_w * (1 + math.cos(x*1.0*math.pi/(train_size-1)))) for x in totoro_rank]
    rn_weight = torch.from_numpy(np.array(rn_weight)).type(torch.FloatTensor)
    rn_weight = rn_weight * data.train_mask.float()
   
    return rn_weight


#loading the processed data
def load_processed_data(opt,data_path,data_name,shuffle_seed = 0, ppr_file=''):
    
    print("\nLoading {} data with shuffle_seed {}".format(data_name,shuffle_seed))

    data_dict = {'cora':'planetoid','citeseer':'planetoid','pubmed':'planetoid',
                'photo':'amazon','computers':'amazon','actor':'Actor',
                'chameleon':'WikipediaNetwork','squirrel':'WikipediaNetwork','arxiv':'ogbn'}    
    target_type = data_dict[data_name]
    if target_type == 'amazon':
        target_dataset = Amazon(data_path, name=data_name)
    elif target_type == 'planetoid':
        target_dataset = Planetoid(data_path, name=data_name)
    elif target_type == 'WikipediaNetwork':
         target_dataset = WikipediaNetwork(root=data_path, name=data_name, geom_gcn_preprocess=True)    
    elif target_type == 'Actor':
        target_dataset = Actor(data_path)
    elif data_name == 'arxiv':
        target_dataset = PygNodePropPredDataset(name='ogbn-arxiv')
    
    target_data=target_dataset[0]
    target_data.num_classes = np.max(target_data.y.numpy())+1

    #the random seed for the dataset splitting
    mask_list = [i for i in range(target_data.num_nodes)]
    random.seed(shuffle_seed)
    random.shuffle(mask_list)

    #none = quantity-balance; step = step quantity-imbalance
    if opt.data_name == 'arxiv':
        train_mask_list,valid_mask_list,test_mask_list,target_data.train_node  = get_split_arxiv(opt,mask_list,target_data.y.numpy(),nclass=target_data.num_classes)
    elif opt.size_imb_type == 'none':
        train_mask_list,valid_mask_list,test_mask_list,target_data.train_node  = get_split(opt,mask_list,target_data.y.numpy(),nclass=target_data.num_classes)
    elif opt.size_imb_type == 'step':
        train_mask_list,valid_mask_list,test_mask_list,target_data.train_node  = get_step_split(opt,mask_list,target_data.y.numpy(),nclass=target_data.num_classes)


    target_data.train_mask = torch.zeros(target_data.num_nodes, dtype=torch.bool)
    target_data.valid_mask = torch.zeros(target_data.num_nodes, dtype=torch.bool)
    target_data.test_mask  = torch.zeros(target_data.num_nodes, dtype=torch.bool)

    target_data.train_mask_list = train_mask_list

    target_data.train_mask[torch.tensor(train_mask_list).long()] = True
    target_data.valid_mask[torch.tensor(valid_mask_list).long()] = True
    target_data.test_mask[torch.tensor(test_mask_list).long()]   = True

    
    # calculating the Personalized PageRank Matrix if not exists.
    if os.path.exists(ppr_file):
        target_data.Pi = torch.load(ppr_file)
    elif opt.data_name == 'arxiv': 
        A = index2sparse(target_data.edge_index, target_data.num_nodes)
        A = A + direct_sparse_eye(target_data.num_nodes)  # Add self-loop
        D = compute_degree_matrix(A, target_data.num_nodes)
        A_normalized = D @ A @ D

        # Initialize PageRank vector
        ppr = torch.ones((target_data.num_nodes, 1)) / target_data.num_nodes
        alpha = opt.pagerank_prob

        for _ in range(40):  # Power iteration
            ppr = (1 - alpha) * A_normalized @ ppr + alpha * (torch.ones((target_data.num_nodes, 1)) / target_data.num_nodes)

        target_data.Pi = ppr        
    else:
        pr_prob = 1 - opt.pagerank_prob
        A = index2dense(target_data.edge_index,target_data.num_nodes)
        A_hat   = A.to(opt.device) + torch.eye(A.size(0)).to(opt.device) # add self-loop
        D       = torch.diag(torch.sum(A_hat,1))
        D       = D.inverse().sqrt()
        A_hat   = torch.mm(torch.mm(D, A_hat), D)
        target_data.Pi = pr_prob * ((torch.eye(A.size(0)).to(opt.device) - (1 - pr_prob) * A_hat).inverse())
        target_data.Pi = target_data.Pi.cpu()
        #torch.save(target_data.Pi,ppr_file)   

    # calculating the ReNode Weight
    gpr_matrix = [] # the class-level influence distribution
    for iter_c in range(target_data.num_classes):
        iter_Pi = target_data.Pi[torch.tensor(target_data.train_node[iter_c]).long()]
        iter_gpr = torch.mean(iter_Pi,dim=0).squeeze()
        gpr_matrix.append(iter_gpr)

    temp_gpr = torch.stack(gpr_matrix,dim=0)
    if temp_gpr.dim() == 1:
        temp_gpr = temp_gpr.unsqueeze(1)
    temp_gpr = temp_gpr.transpose(0,1)
    target_data.gpr = temp_gpr
        
    target_data.rn_weight =  get_renode_weight(opt,target_data) #ReNode Weight 

    
    return target_data

def index2sparse(edge_index, num_nodes):
    # edge_index to sparse format
    row, col = edge_index
    edge_weight = torch.ones(col.size(0), dtype=torch.float32)  # assuming edge weight = 1
    adj_sparse = torch.sparse_coo_tensor(torch.stack([row, col]), edge_weight, (num_nodes, num_nodes))
    return adj_sparse

def direct_sparse_eye(n):
    indices = torch.arange(n)
    indices = torch.stack([indices, indices])
    values = torch.ones(n)
    return torch.sparse_coo_tensor(indices, values, (n, n))

def compute_degree_matrix(A, num_nodes):
    indices = A._indices()
    values = A._values()
    row_indices = indices[0]

    degree = torch.zeros(num_nodes, dtype=values.dtype)

    for idx, value in zip(row_indices, values):
        degree[idx] += value

    degree = degree.pow(-0.5)

    diag_indices = torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)])
    D = torch.sparse_coo_tensor(diag_indices, degree, (num_nodes, num_nodes))

    return D
