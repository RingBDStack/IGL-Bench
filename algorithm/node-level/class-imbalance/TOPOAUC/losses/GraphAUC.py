from sklearn.utils.extmath import softmax
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

INT_MAX=pow(2,31)-100000000

class GAUCLoss(nn.Module):
    def __init__(self, num_classes, num_nodes, adj_matrix, global_effect_matrix,
                 global_perclass_mean_effect_matrix, train_mask, device,
                 weight_sub_dim=64, weight_inter_dim=64, weight_global_dim=64,
                 beta=0.5, gamma=1, is_ner_weight=True, loss_type='ExpGAUC'):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.beta = beta
        self.device = device
        self.loss_type = loss_type
        self.is_ner_weight = is_ner_weight

        self.num_nodes = num_nodes
        self.weight_sub_dim = weight_sub_dim
        self.weight_inter_dim = weight_inter_dim
        self.weight_global_dim = weight_global_dim
        self.adj_matrix = adj_matrix.to(self.device)
        self.global_effect_matrix = global_effect_matrix.to(self.device)
        self.global_perclass_mean_effect_matrix = global_perclass_mean_effect_matrix.to(self.device)
        self.mask = train_mask

        self.I = torch.eye(self.num_nodes, dtype=torch.bool).to(self.device)
        self.adj_self_matrix = self.adj_matrix ^ torch.diag_embed(torch.diag(self.adj_matrix.int())).bool() | self.I

        self.linear_sub = nn.Linear(self.global_effect_matrix.shape[1], self.weight_sub_dim, bias=False).to(self.device)
        self.linear_inter = nn.Linear(self.global_effect_matrix.shape[1], self.weight_inter_dim, bias=False).to(self.device)
        self.linear_global = nn.Linear(self.global_effect_matrix.shape[1], self.weight_global_dim, bias=False).to(self.device)

        nn.init.uniform_(self.linear_sub.weight, a=0.0, b=1.0)
        nn.init.uniform_(self.linear_inter.weight, a=0.0, b=1.0)
        nn.init.uniform_(self.linear_global.weight, a=0.0, b=1.0)

    def forward(self, pred, target, mask, w_values_dict):
        Y = torch.stack([target.eq(i).float() for i in range(self.num_classes)], dim=1).squeeze()
        N = Y.sum(0)  # [classes, 1]
        loss = torch.tensor([0.]).to(self.device)

        self.global_sub = self.linear_sub(self.global_effect_matrix).sum(dim=-1)
        self.global_inter = self.linear_inter(self.global_effect_matrix).sum(dim=-1)
        self.global_global = self.linear_global(self.global_effect_matrix).sum(dim=-1)

        for i in range(self.num_classes):
            for j in range(self.num_classes):
                if i != j:
                    i_pred_pos = pred[Y[:, i].bool(), :][:, i]
                    i_pred_neg = pred[Y[:, j].bool(), :][:, i]
                    i_pred_pos_expand = i_pred_pos.unsqueeze(1).expand(i_pred_pos.shape[0], i_pred_neg.shape[0])
                    i_pred_pos_sub_neg = i_pred_pos_expand - i_pred_neg

                    if self.loss_type == 'SqGAUC':
                        ij_loss = torch.pow(self.gamma - i_pred_pos_sub_neg, 2)
                    elif self.loss_type == 'HingeGAUC':
                        ij_loss = F.relu(self.gamma - i_pred_pos_sub_neg)
                    elif self.loss_type == 'ExpGAUC':
                        ij_loss = torch.exp(-self.gamma * i_pred_pos_sub_neg)

                    if self.is_ner_weight == 0:
                        ij_loss = (1 / (N[i] * N[j]) * ij_loss).sum()
                        loss += ij_loss
                        continue

                    i_pred_pos_index = torch.nonzero(Y[:, i], as_tuple=False).view(-1)
                    i_pred_neg_index = torch.nonzero(Y[:, j], as_tuple=False).view(-1)

                    i_pred_pos_adj = self.adj_matrix[mask][i_pred_pos_index]
                    i_pred_neg_adj = self.adj_matrix[mask][i_pred_neg_index]
                    i_pred_neg_self_adj = self.adj_self_matrix[mask][i_pred_neg_index]

                    i_pred_pos_adj_expand = i_pred_pos_adj.unsqueeze(1).expand(i_pred_pos_adj.shape[0], i_pred_neg_adj.shape[0], i_pred_pos_adj.shape[1])

                    sub_ner = i_pred_pos_adj_expand ^ (i_pred_pos_adj_expand & i_pred_neg_self_adj)
                    inter_ner = i_pred_pos_adj_expand & i_pred_neg_adj

                    max_dim_0=int(np.floor(INT_MAX/sub_ner.shape[2]/sub_ner.shape[1]))
                    if max_dim_0>sub_ner.shape[0]:
                        sub_ner_nonzeros=sub_ner.nonzero(as_tuple=True)
                    else:
                        cal_times=int(np.floor(sub_ner.shape[0]/max_dim_0))
                        if cal_times*max_dim_0<sub_ner.shape[0]:
                            cal_times+=1
                        sub_ner_nonzeros=list(sub_ner[:max_dim_0].nonzero(as_tuple=True))
                        for i in range(1,cal_times):
                            tmp=list(sub_ner[i*max_dim_0:(i+1)*max_dim_0].nonzero(as_tuple=True))
                            for j in range(3):
                                sub_ner_nonzeros[j]=torch.cat((sub_ner_nonzeros[j],tmp[j]))
                    I_sub = torch.stack(sub_ner_nonzeros, dim=0)

                    V_sub = self.global_sub[sub_ner_nonzeros[2]]

                    S_sub = torch.sparse_coo_tensor(I_sub, V_sub, sub_ner.shape).coalesce()
                    vi_sub=S_sub.to_dense()

                    max_dim_0=int(np.floor(INT_MAX/inter_ner.shape[2]/inter_ner.shape[1]))
                    if max_dim_0>inter_ner.shape[0]:
                        inter_ner_nonzeros=inter_ner.nonzero(as_tuple=True)
                    else:
                        cal_times=int(np.floor(inter_ner.shape[0]/max_dim_0))
                        if cal_times*max_dim_0<inter_ner.shape[0]:
                            cal_times+=1
                        inter_ner_nonzeros=list(inter_ner[:max_dim_0].nonzero(as_tuple=True))
                        for i in range(1,cal_times):
                            tmp=list(inter_ner[i*max_dim_0:(i+1)*max_dim_0].nonzero(as_tuple=True))
                            for j in range(3):
                                inter_ner_nonzeros[j]=torch.cat((inter_ner_nonzeros[j],tmp[j]))

                    I_inter=torch.cat([inter_ner_nonzeros[0],inter_ner_nonzeros[1]],0).reshape(2,-1)
                    V_inter=self.global_inter[inter_ner_nonzeros[2]]

                    S_inter=torch.sparse_coo_tensor(I_inter,V_inter,inter_ner.shape[:-1]).coalesce()
                    vi_inter=S_inter.to_dense()
                    
                    vi_inter_expanded = vi_inter.unsqueeze(2)
                    vi_inter_expanded = vi_inter_expanded.expand(-1, -1, vi_sub.size(2))
                    vl_i = torch.sigmoid((1 + vi_sub) / (1 + vi_inter_expanded))
                    #vl_i=torch.sigmoid((1+vi_sub)/(1+vi_inter))

                    i_nonzeros=i_pred_pos_adj.nonzero(as_tuple=True)
                    I_yi=i_nonzeros[0].reshape(1,-1)
                    V_yi=self.global_global[i_nonzeros[1]]
                    S_yi=torch.sparse_coo_tensor(I_yi,V_yi,i_pred_pos_adj.shape[:-1]).coalesce()
                    vi_g=S_yi.to_dense()
                    # vi_g=torch.matmul(i_pred_pos_adj.float(),self.global_global).sum(dim=-1)

                    non_i_nonzeros=i_pred_neg_adj.nonzero(as_tuple=True)
                    I_non_yi=non_i_nonzeros[0].reshape(1,-1)
                    V_non_yi=self.global_global[non_i_nonzeros[1]]
                    S_non_yi=torch.sparse_coo_tensor(I_non_yi,V_non_yi,i_pred_neg_adj.shape[:-1]).coalesce()
                    v_non_i_g=S_non_yi.to_dense()
                    vg_i=torch.sigmoid(vi_g.unsqueeze(1).expand(vi_g.shape[0],v_non_i_g.shape[0])-v_non_i_g)
                    v_i=1-vl_i
                    
                    ij_loss_expanded = ij_loss.unsqueeze(2).expand(-1, -1, v_i.shape[2])
                    ij_loss=(1/(N[i]*N[j])*v_i*ij_loss_expanded).sum()

                    cur_len=inter_ner.nonzero().shape[0]
                    inter_ner = inter_ner.to('cpu')
                    pos_idx=mask.nonzero().view(-1)[inter_ner.nonzero()[:,0]].numpy()
                    neg_idx=mask.nonzero().view(-1)[inter_ner.nonzero()[:,1]].numpy()
                    #values=v_i[inter_ner.nonzero()[:,0],inter_ner.nonzero()[:,1]].cpu().detach().numpy()
                    batch_size = 10000  # 根据实际情况调整这个值
                    nonzero_indices = inter_ner.nonzero()
                    total_batches = (nonzero_indices.size(0) + batch_size - 1) // batch_size

                    # 存储计算的值，供之后使用
                    values_list = []

                    for batch_idx in range(total_batches):
                        start_idx = batch_idx * batch_size
                        end_idx = min(start_idx + batch_size, nonzero_indices.size(0))
                        batch_indices = nonzero_indices[start_idx:end_idx]

                        # 从 v_i 中提取对应的元素
                        batch_values = v_i[batch_indices[:, 0], batch_indices[:, 1]].cpu().detach().numpy()
                        values_list.append(batch_values)

                    # 合并所有批次处理的结果
                    values = np.concatenate(values_list)

                    for cur_pos in range(len(values)):  # 注意这里改为迭代 values 的长度
                        if pos_idx[cur_pos] not in w_values_dict.keys():
                            w_values_dict[pos_idx[cur_pos]] = {}
                        if neg_idx[cur_pos] not in w_values_dict[pos_idx[cur_pos]].keys():
                            w_values_dict[pos_idx[cur_pos]][neg_idx[cur_pos]] = []
                        w_values_dict[pos_idx[cur_pos]][neg_idx[cur_pos]].append(values[cur_pos])

                    loss += ij_loss
                
        return loss
'''
    def forward(self, pred, target, mask, w_values_dict):
        Y = torch.stack([target.eq(i).float() for i in range(self.num_classes)], dim=1).squeeze()
        N = Y.sum(0)  # [classes, 1]
        loss = torch.tensor([0.]).to(self.device)

        self.global_sub = self.linear_sub(self.global_effect_matrix).sum(dim=-1)
        self.global_inter = self.linear_inter(self.global_effect_matrix).sum(dim=-1)
        self.global_global = self.linear_global(self.global_effect_matrix).sum(dim=-1)

        for i in range(self.num_classes):
            for j in range(self.num_classes):
                if i != j:
                    i_pred_pos = pred[Y[:, i].bool(), :][:, i]
                    i_pred_neg = pred[Y[:, j].bool(), :][:, i]
                    i_pred_pos_expand = i_pred_pos.unsqueeze(1).expand(i_pred_pos.shape[0], i_pred_neg.shape[0])
                    i_pred_pos_sub_neg = i_pred_pos_expand - i_pred_neg

                    if self.loss_type == 'SqGAUC':
                        ij_loss = torch.pow(self.gamma - i_pred_pos_sub_neg, 2)
                    elif self.loss_type == 'HingeGAUC':
                        ij_loss = F.relu(self.gamma - i_pred_pos_sub_neg)
                    elif self.loss_type == 'ExpGAUC':
                        ij_loss = torch.exp(-self.gamma * i_pred_pos_sub_neg)

                    if self.is_ner_weight == 0:
                        ij_loss = (1 / (N[i] * N[j]) * ij_loss).sum()
                        loss += ij_loss
                        continue

                    i_pred_pos_index = torch.nonzero(Y[:, i], as_tuple=False).view(-1)
                    i_pred_neg_index = torch.nonzero(Y[:, j], as_tuple=False).view(-1)

                    i_pred_pos_adj = self.adj_matrix[mask][i_pred_pos_index]
                    i_pred_neg_adj = self.adj_matrix[mask][i_pred_neg_index]
                    i_pred_neg_self_adj = self.adj_self_matrix[mask][i_pred_neg_index]

                    i_pred_pos_adj_expand = i_pred_pos_adj.unsqueeze(1).expand(i_pred_pos_adj.shape[0], i_pred_neg_adj.shape[0], i_pred_pos_adj.shape[1])

                    sub_ner = i_pred_pos_adj_expand ^ (i_pred_pos_adj_expand & i_pred_neg_self_adj)
                    inter_ner = i_pred_pos_adj_expand & i_pred_neg_adj

                    max_dim_0=int(np.floor(INT_MAX/sub_ner.shape[2]/sub_ner.shape[1]))
                    if max_dim_0>sub_ner.shape[0]:
                        sub_ner_nonzeros=sub_ner.nonzero(as_tuple=True)
                    else:
                        cal_times=int(np.floor(sub_ner.shape[0]/max_dim_0))
                        if cal_times*max_dim_0<sub_ner.shape[0]:
                            cal_times+=1
                        sub_ner_nonzeros=list(sub_ner[:max_dim_0].nonzero(as_tuple=True))
                        for i in range(1,cal_times):
                            tmp=list(sub_ner[i*max_dim_0:(i+1)*max_dim_0].nonzero(as_tuple=True))
                            for j in range(3):
                                sub_ner_nonzeros[j]=torch.cat((sub_ner_nonzeros[j],tmp[j]))
                    I_sub = torch.stack(sub_ner_nonzeros, dim=0)

                    V_sub = self.global_sub[sub_ner_nonzeros[2]]

                    S_sub = torch.sparse_coo_tensor(I_sub, V_sub, sub_ner.shape).coalesce()
                    vi_sub=S_sub.to_dense()

                    max_dim_0=int(np.floor(INT_MAX/inter_ner.shape[2]/inter_ner.shape[1]))
                    if max_dim_0>inter_ner.shape[0]:
                        inter_ner_nonzeros=inter_ner.nonzero(as_tuple=True)
                    else:
                        cal_times=int(np.floor(inter_ner.shape[0]/max_dim_0))
                        if cal_times*max_dim_0<inter_ner.shape[0]:
                            cal_times+=1
                        inter_ner_nonzeros=list(inter_ner[:max_dim_0].nonzero(as_tuple=True))
                        for i in range(1,cal_times):
                            tmp=list(inter_ner[i*max_dim_0:(i+1)*max_dim_0].nonzero(as_tuple=True))
                            for j in range(3):
                                inter_ner_nonzeros[j]=torch.cat((inter_ner_nonzeros[j],tmp[j]))

                    I_inter=torch.cat([inter_ner_nonzeros[0],inter_ner_nonzeros[1]],0).reshape(2,-1)
                    V_inter=self.global_inter[inter_ner_nonzeros[2]]

                    S_inter=torch.sparse_coo_tensor(I_inter,V_inter,inter_ner.shape[:-1]).coalesce()
                    vi_inter=S_inter.to_dense()
                    
                    vi_inter_expanded = vi_inter.unsqueeze(2)
                    vi_inter_expanded = vi_inter_expanded.expand(-1, -1, vi_sub.size(2))
                    vl_i = torch.sigmoid((1 + vi_sub) / (1 + vi_inter_expanded))
                    #vl_i=torch.sigmoid((1+vi_sub)/(1+vi_inter))

                    i_nonzeros=i_pred_pos_adj.nonzero(as_tuple=True)
                    I_yi=i_nonzeros[0].reshape(1,-1)
                    V_yi=self.global_global[i_nonzeros[1]]
                    S_yi=torch.sparse_coo_tensor(I_yi,V_yi,i_pred_pos_adj.shape[:-1]).coalesce()
                    vi_g=S_yi.to_dense()
                    # vi_g=torch.matmul(i_pred_pos_adj.float(),self.global_global).sum(dim=-1)

                    non_i_nonzeros=i_pred_neg_adj.nonzero(as_tuple=True)
                    I_non_yi=non_i_nonzeros[0].reshape(1,-1)
                    V_non_yi=self.global_global[non_i_nonzeros[1]]
                    S_non_yi=torch.sparse_coo_tensor(I_non_yi,V_non_yi,i_pred_neg_adj.shape[:-1]).coalesce()
                    v_non_i_g=S_non_yi.to_dense()
                    vg_i=torch.sigmoid(vi_g.unsqueeze(1).expand(vi_g.shape[0],v_non_i_g.shape[0])-v_non_i_g)
                    v_i=1-vl_i
                    
                    ij_loss_expanded = ij_loss.unsqueeze(2).expand(-1, -1, v_i.shape[2])
                    ij_loss=(1/(N[i]*N[j])*v_i*ij_loss_expanded).sum()

                    cur_len=inter_ner.nonzero().shape[0]
                    inter_ner = inter_ner.to('cpu')
                    pos_idx=mask.nonzero().view(-1)[inter_ner.nonzero()[:,0]].numpy()
                    neg_idx=mask.nonzero().view(-1)[inter_ner.nonzero()[:,1]].numpy()
                    values=v_i[inter_ner.nonzero()[:,0],inter_ner.nonzero()[:,1]].cpu().detach().numpy()
                    for cur_pos in range(cur_len):
                        if pos_idx[cur_pos] not in w_values_dict.keys():
                            w_values_dict[pos_idx[cur_pos]]={}
                        if neg_idx[cur_pos] not in w_values_dict[pos_idx[cur_pos]].keys():
                            w_values_dict[pos_idx[cur_pos]][neg_idx[cur_pos]]=[]
                        w_values_dict[pos_idx[cur_pos]][neg_idx[cur_pos]].append(values[cur_pos])

                    loss+=ij_loss
                
        return loss
    '''
class ELossFN(nn.Module):
    def __init__(self, num_classes, num_nodes, adj_matrix, global_effect_matrix,
                 global_perclass_mean_effect_matrix, train_mask, device,
                 weight_sub_dim=64, weight_inter_dim=64, weight_global_dim=64,
                 beta=0.5, gamma=1, is_ner_weight=True, loss_type='ExpGAUC',per=1e-3):
        super(ELossFN, self).__init__()

        self.num_classes = num_classes
        self.gamma = gamma
        self.beta = beta
        self.loss_type = loss_type
        self.is_ner_weight = is_ner_weight
        self.mask = train_mask
        self.sceloss= nn.CrossEntropyLoss()
        self.per = per
        self.l2_loss = nn.MSELoss()
        self.device = device

        self.num_nodes = num_nodes
        self.weight_sub_dim = weight_sub_dim
        self.weight_inter_dim = weight_inter_dim
        self.weight_global_dim = weight_global_dim
        self.adj_matrix = adj_matrix

        # Converting global effect matrix according to mask
        self.global_effect_matrix = torch.tensor(global_effect_matrix)[torch.tensor(train_mask)]
        self.global_perclass_mean_effect_matrix = torch.tensor(global_perclass_mean_effect_matrix)

        # Creating an identity matrix of shape (num_nodes, num_nodes)
        self.I = torch.eye(num_nodes, dtype=torch.bool,device=device)

        # Creating self adjacency matrix
        hou = torch.diag(torch.diagonal(adj_matrix))
        qian = adj_matrix ^ hou
        self.adj_self_matrix = qian | self.I

        print("GAUC, OK")

    def gem_cut(self, gem, mask):

        gem = gem.t()
        return gem[mask].t()

    def get_pred(self, preds, mask):

        return preds[mask]

    def get_tem_label(self, label, mask):

        return label[mask]

    def get_label(self, tem_label):

        return torch.nonzero(tem_label, as_tuple=True)   
    
    def show(self, item):

        print(item, type(item))
        print(item.shape)

    def nonzero_tuple(self, inp):

        return tuple(inp.nonzero(as_tuple=True))   
    
    def forward(self, preds, labels, mask, w_values_dict):
        mask_tensor = mask.clone().detach()
        mask_tensor = mask_tensor.bool()  # 转换为布尔型


        pred = self.get_pred(preds, mask_tensor)
        label = self.get_pred(labels, mask_tensor)


        # Create one-hot encoding 
        Y = F.one_hot(label, num_classes=self.num_classes).float()
        N = Y.sum(dim=0)

        # Calculate losses
        loss = self.sceloss(preds, labels).item()
        #loss = torch.tensor([0.]).to(self.device)
        
        for i in range(self.num_classes):  
            for j in range(self.num_classes):
                if i != j:
                    i_pred_pos = pred[Y[:, i].bool(), i]
                    i_pred_neg = pred[Y[:, j].bool(), i]

                    # Expand i_pred_pos to match dimensions of i_pred_neg
                    i_pred_pos_expand = i_pred_pos.unsqueeze(1).expand(-1, i_pred_neg.size(0))
                    i_pred_pos_sub_neg = i_pred_pos_expand - i_pred_neg

                    ij_loss = torch.exp(-self.gamma * i_pred_pos_sub_neg)

                    # Finding indices where Y[:,i] and Y[:,j] are 1
                    i_pred_pos_index = Y[:, i].nonzero(as_tuple=True)[0]
                    i_pred_neg_index = Y[:, j].nonzero(as_tuple=True)[0]

                    # Adjacency matrix operations
                    i_pred_pos_adj = self.adj_matrix[mask_tensor][i_pred_pos_index]
                    i_pred_neg_adj = self.adj_matrix[mask_tensor][i_pred_neg_index]
                    i_pred_neg_self_adj = self.adj_self_matrix[mask_tensor][i_pred_neg_index]

                    # Expand i_pred_pos_adj to perform logical operations
                    i_pred_pos_adj_expand = i_pred_pos_adj.unsqueeze(1).expand(-1, i_pred_neg_adj.size(0), -1)

                    # Logical XOR and AND operations
                    sub_ner = torch.logical_xor(i_pred_pos_adj_expand, torch.logical_and(i_pred_pos_adj_expand, i_pred_neg_self_adj))
                    inter_ner = torch.logical_and(i_pred_pos_adj_expand, i_pred_neg_adj)

                    # Count nonzero elements
                    sub_ner_nonzero = self.nonzero_tuple(sub_ner)
                    inter_ner_nonzero = self.nonzero_tuple(inter_ner)
                    
                    # Create sparse matrices
                    if sub_ner_nonzero[0].numel() > 0 and inter_ner_nonzero[0].numel() > 0:
                        I_sub = torch.stack(sub_ner_nonzero[:2]).t()
                        V_sub = torch.sigmoid(sub_ner[sub_ner_nonzero].float())
                        vi_sub = torch.sparse_coo_tensor(I_sub.t(), V_sub, i_pred_pos_sub_neg.shape).to_dense()

                        I_inter = torch.stack(inter_ner_nonzero[:2]).t()
                        V_inter = torch.sigmoid(inter_ner[inter_ner_nonzero].float())
                        vi_inter = torch.sparse_coo_tensor(I_inter.t(), V_inter, i_pred_pos_sub_neg.shape).to_dense()
                        #ij_loss = (1 / (N[i] * N[j]) * v_i * ij_loss) * self.per
                        # Compute final components for loss
                        vl_i = torch.sigmoid((1 + vi_sub) / (1 + vi_inter))
                        v_i = 1 - vl_i
                        ij_loss = ij_loss * (1 / (N[i] * N[j])) * v_i
                        ij_loss = ij_loss.sum() * self.per
                        loss += ij_loss

        return loss