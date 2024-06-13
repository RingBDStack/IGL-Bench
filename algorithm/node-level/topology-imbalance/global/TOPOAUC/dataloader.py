import random
import codecs
import copy
import math
import os,sys

import torch
import torch.nn.functional as F
import numpy as np
from utils import index2dense
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Amazon,Actor,WikipediaNetwork
from ogb.nodeproppred import PygNodePropPredDataset

#access a quantity-balanced training set: each class has the same training size train_each
def get_split(opt,all_idx,all_label,nclass):
    train_each = (len(all_idx)*0.1) // nclass 
    valid_each = (len(all_idx)*0.1) // nclass 

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

#return the ReNode Weight
def get_renode_weight(opts,data):

    ppr_matrix = data.Pi  #personlized pagerank
    gpr_matrix = torch.tensor(data.gpr).float() #class-accumulated personlized pagerank

    base_w  = opts.rn_base_weight
    scale_w = opts.rn_scale_weight
    nnode = ppr_matrix.size(0)
    unlabel_mask = data.train_mask.int().ne(1)#unlabled node


    #computing the Totoro values for labeled nodes
    gpr_sum = torch.sum(gpr_matrix,dim=1)
    gpr_rn  = gpr_sum.unsqueeze(1) - gpr_matrix
    rn_matrix = torch.mm(ppr_matrix,gpr_rn)

    label_matrix = F.one_hot(data.y,gpr_matrix.size(1)).float() 
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

#access a quantity-imbalanced training set; the training set follows the step distribution.
def get_step_split(opts,all_nodes_idx,all_label,nclass=7):

    base_valid_each = opts.valid_each

    imb_ratio = opts.imb_ratio
    #head_list = opts.head_list if len(opts.head_list)>0 else [i for i in range(nclass//2)]
    head_list = [i for i in range(nclass//2)]

    all_class_list = [i for i in range(nclass)]
    tail_list = list(set(all_class_list) - set(head_list))

    h_num = len(head_list)
    t_num = len(tail_list)

    base_train_each = int( len(all_nodes_idx) * opts.labeling_ratio / (t_num + h_num * imb_ratio) )

    idx2train,idx2valid = {},{}

    total_train_size = 0
    total_valid_size = 0

    for i_h in head_list: 
        idx2train[i_h] = int(base_train_each * imb_ratio)
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

    for iter1 in all_nodes_idx:
        iter_label = all_label[iter1]
        if train_list[iter_label] < idx2train[iter_label]:
            train_list[iter_label]+=1
            train_node[iter_label].append(iter1)
            train_idx.append(iter1)

        if sum(train_list)==total_train_size:break

    assert sum(train_list)==total_train_size

    after_train_idx = list(set(all_nodes_idx)-set(train_idx))

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


def max_degree_of_train_nodes(data,log):
    max_degree=-1
    degrees={}
    edges=data.edge_index.numpy().tolist()
    edges_nums=data.edge_index.shape[1]
    for i in range(edges_nums):
        ni=edges[0][i]
        nj=edges[1][i]
        if ni not in degrees.keys():
            degrees[ni]=0
        if nj not in degrees.keys():
            degrees[nj]=0
        degrees[ni]+=1
        degrees[nj]+=1
    for cur_train_node in data.train_mask_list:
        if cur_train_node in degrees.keys():
            if degrees[cur_train_node]>max_degree:
                max_degree=degrees[cur_train_node]
    log.info("max degree of train nodes: {}".format(max_degree))
    return max_degree

#loading the processed data
def load_processed_data(opts,log,data_path,data_name,shuffle_seed = 0, gem_file=''):
    log.info("Loading {} data with shuffle_seed {}".format(data_name,shuffle_seed))
    data_dict = {'cora':'planetoid','citeseer':'planetoid','pubmed':'planetoid',
                'photo':'amazon','computers':'amazon','actor':'Actor',
                'chameleon':'WikipediaNetwork','squirrel':'WikipediaNetwork','arxiv':'ogbn'}
    target_type = data_dict[data_name]
    if target_type == 'amazon':
        target_dataset = Amazon(root=data_path+data_name, name=data_name)
    elif target_type == 'planetoid':
        target_dataset = Planetoid(data_path, name=data_name)
    elif target_type == 'WikipediaNetwork':
         target_dataset = WikipediaNetwork(root=data_path, name=data_name, geom_gcn_preprocess=True)    
    elif target_type == 'Actor':
        target_dataset = Actor(data_path+data_name)
    elif data_name == 'arxiv':
        target_dataset = PygNodePropPredDataset(name='ogbn-arxiv')
    
    target_data=target_dataset[0]
    target_data.num_classes = np.max(target_data.y.numpy())+1

    node_idx_list = [i for i in range(target_data.num_nodes)]
    random.seed(shuffle_seed)
    random.shuffle(node_idx_list)

    if opts.is_imb:
        train_mask_list,valid_mask_list,test_mask_list,target_data.train_node  = get_step_split(opts,node_idx_list,target_data.y.numpy(),nclass=target_data.num_classes)
    else:
        train_mask_list,valid_mask_list,test_mask_list,target_data.train_node  = get_split(opts,node_idx_list,target_data.y.numpy(),nclass=target_data.num_classes)

    target_data.train_mask = torch.zeros(target_data.num_nodes, dtype=torch.bool)
    target_data.valid_mask = torch.zeros(target_data.num_nodes, dtype=torch.bool)
    target_data.test_mask  = torch.zeros(target_data.num_nodes, dtype=torch.bool)

    target_data.train_mask_list = train_mask_list

    target_data.train_mask[torch.tensor(train_mask_list).long()] = True
    target_data.valid_mask[torch.tensor(valid_mask_list).long()] = True
    target_data.test_mask[torch.tensor(test_mask_list).long()]   = True

    target_data.train_max_degree =max_degree_of_train_nodes(target_data,log)
    label_counts = {}
    min_class_num = 0
    y_label=target_data.y.numpy()
    for cur_train_index in train_mask_list:
        cur_y=y_label[cur_train_index]
        if cur_y not in label_counts.keys():
            label_counts[cur_y]=[cur_train_index]
        else:
            label_counts[cur_y].append(cur_train_index)
    target_data.train_label_dict=label_counts
    label_counts_lens=[len(label_counts[l]) for l in range(target_data.num_classes)]
    target_data.label_counts_lens=label_counts_lens
    target_data.train_min_class_num=np.min(label_counts_lens)
    target_data.train_max_class_num=np.max(label_counts_lens)
    target_data.train_imb_r=np.max(label_counts_lens)//np.min(label_counts_lens)

    mini_mask_len=10
    mini_mask_nums=int(np.ceil(target_data.train_min_class_num/mini_mask_len))
    mini_train_mask_dict={}
    for i in range(mini_mask_nums):
        mini_train_mask_dict[i]=[]
        for cur_key in range(target_data.num_classes):
            cur_key_len=mini_mask_len
            if len(target_data.train_label_dict[cur_key])==target_data.train_max_class_num:
                cur_key_len=cur_key_len*target_data.train_imb_r
            mini_train_mask_dict[i].append(target_data.train_label_dict[cur_key][i*cur_key_len:(i+1)*cur_key_len])
    mini_train_mask_list=[]
    for cur_class in range(target_data.num_classes-1):
        for cur_pos_mini in range(mini_mask_nums):
            cur_pos_mini_list=mini_train_mask_dict[cur_pos_mini][cur_class]
            for cur_neg_mini in range(mini_mask_nums):
                if cur_class and cur_pos_mini==cur_neg_mini:
                    continue
                cur_neg_mini_list_1=mini_train_mask_dict[cur_neg_mini][cur_class+1:]
                cur_neg_mini_list=[]
                for pos in range(len(cur_neg_mini_list_1)):
                    cur_neg_mini_list.extend(cur_neg_mini_list_1[pos])
                cur_neg_mini_list.extend(cur_pos_mini_list)
                mini_train_mask_list.append(cur_neg_mini_list)

    target_data.mini_train_mask_list=mini_train_mask_list
    pr_prob = opts.pagerank_prob
    A = index2dense(target_data.edge_index,target_data.num_nodes)
    A_hat   = A.to(opts.device) + torch.eye(A.size(0)).to(opts.device) # add self-loop
    D       = torch.diag(torch.sum(A_hat,1))
    D       = D.inverse().sqrt()
    A_hat   = torch.mm(torch.mm(D, A_hat), D)
    target_data.Pi = pr_prob * ((torch.eye(A.size(0)).to(opts.device) - (1 - pr_prob) * A_hat).inverse())
    target_data.Pi = target_data.Pi.cpu()
    A_hat=A_hat.cpu()
    D=D.cpu()
    torch.cuda.empty_cache()

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
    
    target_data.rn_weight =  get_renode_weight(opts,target_data) #ReNode Weight 

    return target_data

def preprocessDataDRGCN(target_data,log):
    x=csr_matrix(target_data.x,dtype=np.float64)
    adj_matrix=index2dense(target_data.edge_index,target_data.num_nodes)
    A_hat= adj_matrix+ torch.eye(adj_matrix.size(0)) # add self-loop
    D = torch.diag(torch.sum(A_hat,1))
    D = D.inverse().sqrt()
    A_hat = torch.mm(torch.mm(D, A_hat), D)
    adj_matrix=csr_matrix(adj_matrix.numpy(),dtype=np.float64)
    adj_norm=csr_matrix(A_hat.numpy(),dtype=np.float64)

    y_onehot=F.one_hot(target_data.y,target_data.num_classes)
    y_onehot=csr_matrix(y_onehot.numpy(),dtype=np.float64)

    train_indexes=torch.nonzero(target_data.train_mask).view(-1).numpy()
    validation_indexes=torch.nonzero(target_data.val_mask).view(-1).numpy()
    test_indexes=torch.nonzero(target_data.test_mask).view(-1).numpy()

    label_counts = {}
    balance_num = 0
    y_label=target_data.y.numpy()
    for cur_train_index in train_indexes:
        cur_y=y_label[cur_train_index]
        if cur_y not in label_counts.keys():
            label_counts[cur_y]=[cur_train_index]
        else:
            label_counts[cur_y].append(cur_train_index)
        if balance_num < len(label_counts[cur_y]):
                balance_num = len(label_counts[cur_y])
    
    label_dist = [(k,len(label_counts[k])) for k in sorted(label_counts.keys())]      
    log.info('label_distribution: {}'.format(label_dist))
    log.info('balance_num: {}'.format(balance_num))
    # Sample real nodes for training the GAN model
    real_node_sequence = []
    real_gan_nodes = []
    generated_gan_nodes = []
    # add all labeled nodes for training the gan
    for lab in label_counts.keys():
        nodes = label_counts[lab]
        for no in nodes:
            real_gan_nodes.append([no,str(lab)])
            real_node_sequence.append(no)

        balance_differ = balance_num - len(nodes)
        for i in range(balance_differ):
            idx = random.randint(0, len(nodes)-1)
            real_gan_nodes.append([nodes[idx],str(lab)])
            real_node_sequence.append(nodes[idx])
            generated_gan_nodes.append([nodes[idx],str(lab)])

    # shuffle the training samples
    shuffle_indices = np.random.permutation(np.arange(len(real_gan_nodes)))
    real_gan_nodes = [real_gan_nodes[i] for i in shuffle_indices]
    real_node_sequence = [real_node_sequence[i] for i in shuffle_indices]
    log.info('real_gan_nodes: {}'.format(real_gan_nodes) )
    log.info('real_node_sequence: {}'.format(real_node_sequence))
    
    
    # Collect all neighborhood and identically labeled nodes for real nodes
    ori_adj_matrix=index2dense(target_data.edge_index,target_data.num_nodes)
    adjlist = {}
    all_neighbor_nodes = []
    for cur_node in real_node_sequence:
        cur_node_ners=torch.nonzero(ori_adj_matrix[cur_node]).view(-1).numpy().tolist()
        if cur_node not in adjlist.keys():
            adjlist[cur_node]=cur_node_ners
        for cur_ner in cur_node_ners:
            if cur_ner not in all_neighbor_nodes:
                all_neighbor_nodes.append(cur_ner)
    
    real_node_num = len(real_node_sequence)
    real_neighbor_num = len(all_neighbor_nodes)
    adj_neighbor = np.zeros([real_node_num, real_neighbor_num])
    
    for i in range(real_node_num):
        for j in range(real_neighbor_num):
            if all_neighbor_nodes[j] in adjlist[real_node_sequence[i]]:
                adj_neighbor[i][j] = 1
    
    log.info(adj_neighbor[0:1])
    log.info(adj_neighbor.shape)

    return x, adj_matrix, adj_norm, y_onehot, train_indexes, test_indexes, validation_indexes, real_gan_nodes, generated_gan_nodes, adj_neighbor, all_neighbor_nodes


def preprocessDataGraphSMOTE(target_data,log):
    features=target_data.x
    features = sp.csr_matrix(features, dtype=np.float32)
    #norm feature
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    features = torch.FloatTensor(np.array(features.todense()))
    labels=target_data.y
    labels = torch.LongTensor(labels)

    edges=target_data.edge_index.numpy()
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    sparse_mx = adj.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    adj=torch.sparse.FloatTensor(indices, values, shape)

    idx_train=torch.nonzero(target_data.train_mask).view(-1)
    idx_val=torch.nonzero(target_data.val_mask).view(-1)
    idx_test=torch.nonzero(target_data.test_mask).view(-1)

    head_list = [i for i in range(target_data.num_classes//2)]
    all_class_list = [i for i in range(target_data.num_classes)]
    tail_list = list(set(all_class_list) - set(head_list))
    h_num = len(head_list)
    im_class_num = len(tail_list)

    return adj,features,labels,idx_train, idx_val, idx_test, im_class_num

def index2sparse(edge_index, num_nodes):
    # edge_index to sparse format
    row, col = edge_index
    edge_weight = torch.ones(col.size(0), dtype=torch.float32)  # assuming edge weight = 1
    adj_sparse = torch.sparse_coo_tensor(torch.stack([row, col]), edge_weight, (num_nodes, num_nodes))
    return adj_sparse
def direct_sparse_eye(n):
    # 构造单位矩阵的索引（对角线位置）
    indices = torch.arange(n)
    indices = torch.stack([indices, indices])
    # 所有对角线元素都是1
    values = torch.ones(n)
    # 创建稀疏矩阵
    return torch.sparse_coo_tensor(indices, values, (n, n))

def compute_degree_matrix(A, num_nodes):
    # A 是一个稀疏邻接矩阵
    indices = A._indices()
    values = A._values()
    row_indices = indices[0]

    # 初始化度数为0
    degree = torch.zeros(num_nodes, dtype=values.dtype)

    # 对每个行索引位置累加其对应的值
    for idx, value in zip(row_indices, values):
        degree[idx] += value

    # 计算D^-0.5
    degree = degree.pow(-0.5)

    # 构建稀疏度矩阵D
    diag_indices = torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)])
    D = torch.sparse_coo_tensor(diag_indices, degree, (num_nodes, num_nodes))

    return D