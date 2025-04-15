import os
import torch
import torch.nn.functional as F
import math
import numpy as np
from IGL_Bench.algorithm.ReNode.util import index2sparse, direct_sparse_eye, compute_degree_matrix, index2dense 

def get_renode_weight(config, data):
    ppr_matrix = data.Pi  
    gpr_matrix = torch.tensor(data.gpr).float()  

    base_w = config.rn_base_weight
    scale_w = config.rn_scale_weight
    nnode = ppr_matrix.size(0)
    unlabel_mask = data.train_mask.int().ne(1)  

    gpr_sum = torch.sum(gpr_matrix, dim=1)
    gpr_rn = gpr_sum.unsqueeze(1) - gpr_matrix
    rn_matrix = torch.mm(ppr_matrix, gpr_rn)

    labels = data.y.squeeze() 
    label_matrix = F.one_hot(labels, gpr_matrix.size(1)).float()
    label_matrix[unlabel_mask] = 0

    rn_matrix = torch.sum(rn_matrix * label_matrix, dim=1)
    rn_matrix[unlabel_mask] = rn_matrix.max() + 99  

    train_size = torch.sum(data.train_mask.int()).item()
    totoro_list = rn_matrix.tolist()
    id2totoro = {i: totoro_list[i] for i in range(len(totoro_list))}
    sorted_totoro = sorted(id2totoro.items(), key=lambda x: x[1], reverse=False)
    id2rank = {sorted_totoro[i][0]: i for i in range(nnode)}
    totoro_rank = [id2rank[i] for i in range(nnode)]

    rn_weight = [(base_w + 0.5 * scale_w * (1 + math.cos(x * 1.0 * math.pi / (train_size - 1)))) for x in totoro_rank]
    rn_weight = torch.from_numpy(np.array(rn_weight)).type(torch.FloatTensor)
    rn_weight = rn_weight * data.train_mask.float()

    return rn_weight

def compute_rn_weight(dataset, config):
    target_data = dataset
    
    train_index = dataset.train_index
    num_classes = dataset.y.numpy().max().item() + 1
    train_node = [[] for _ in range(num_classes)]
    num_classes = torch.max(target_data.y).item() + 1
    for class_id in range(num_classes):
        class_mask = target_data.y.eq(class_id) 
        for idx in target_data.train_index:
            if class_mask[idx]:  
                train_node[class_id].append(idx)
        
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ppr_file = os.path.join(current_dir, '../../../PPR_file', 
                               f"{target_data.data_name}_pagerank.pt")

    if os.path.exists(ppr_file):
        target_data.Pi = torch.load(ppr_file)
    elif dataset.data_name == 'ogbn-arxiv': 
        A = index2sparse(target_data.edge_index, target_data.num_nodes)
        A = A + direct_sparse_eye(target_data.num_nodes)  # Add self-loop
        D = compute_degree_matrix(A, target_data.num_nodes)
        A_normalized = D @ A @ D

        ppr = torch.ones((target_data.num_nodes, 1)) / target_data.num_nodes
        alpha = config.pagerank_prob

        for _ in range(40):  # Power iteration
            ppr = (1 - alpha) * A_normalized @ ppr + alpha * (torch.ones((target_data.num_nodes, 1)) / target_data.num_nodes)

        target_data.Pi = ppr        
    else:
        pr_prob = 1 - config.pagerank_prob
        A = index2dense(target_data.edge_index, target_data.num_nodes)
        A_hat = A + torch.eye(A.size(0))  # Add self-loop
        D = torch.diag(torch.sum(A_hat, 1))
        D = D.inverse().sqrt()
        A_hat = torch.mm(torch.mm(D, A_hat), D)
        target_data.Pi = pr_prob * ((torch.eye(A.size(0)) - (1 - pr_prob) * A_hat).inverse())
        target_data.Pi = target_data.Pi.cpu()

    gpr_matrix = []  
    for iter_c in range(num_classes):
        iter_Pi = target_data.Pi[torch.tensor(train_node[iter_c]).long()]
        iter_gpr = torch.mean(iter_Pi, dim=0).squeeze()
        gpr_matrix.append(iter_gpr)

    temp_gpr = torch.stack(gpr_matrix, dim=0)
    if temp_gpr.dim() == 1:
        temp_gpr = temp_gpr.unsqueeze(1)
    temp_gpr = temp_gpr.transpose(0, 1)
    target_data.gpr = temp_gpr

    rn_weight = get_renode_weight(config, target_data)

    return rn_weight
