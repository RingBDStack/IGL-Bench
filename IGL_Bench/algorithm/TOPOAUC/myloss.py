import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.I = torch.eye(num_nodes, dtype=torch.bool,device=adj_matrix.device)

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
    
    def forward(self, preds, labels, mask):
        self.adj_self_matrix = self.adj_self_matrix.to(self.device)
        self.adj_matrix = self.adj_matrix.to(self.device)
        
        mask_tensor = mask.clone().detach()
        mask_tensor = mask_tensor.bool()  


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