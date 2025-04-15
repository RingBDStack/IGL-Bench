import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphLearner(nn.Module):
    def __init__(self, input_size, hidden_size, n_nodes, n_class, n_anchors, topk=None, epsilon=None, n_pers=16, device=None):
        super(GraphLearner, self).__init__()
        self.n_nodes = n_nodes
        self.n_class = n_class
        self.n_anchors = n_anchors
        self.topk = topk
        self.epsilon = epsilon
        self.device = device
        self.input_size=input_size

        self.weight_tensor = torch.Tensor(n_pers, input_size)
        self.weight_tensor = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor))
        
        self.weight_tensor_for_pe = torch.Tensor(self.n_anchors, hidden_size)
        self.weight_tensor_for_pe = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor_for_pe))

    def forward(self, context, position_encoding, gpr_rank, position_flag, ctx_mask=None):
        expand_weight_tensor = self.weight_tensor.unsqueeze(1)
        if len(context.shape) == 3:
            expand_weight_tensor = expand_weight_tensor.unsqueeze(1)
        context_fc = context.unsqueeze(0) * expand_weight_tensor
        context_norm = F.normalize(context_fc, p=2, dim=-1)
        attention = torch.bmm(context_norm, context_norm.transpose(-1, -2)).mean(0)

        if position_flag == 1:
            pe_fc = torch.mm(position_encoding, self.weight_tensor_for_pe)
            pe_attention = torch.mm(pe_fc, pe_fc.transpose(-1, -2))
            try:
                attention = (attention * 0.5 + pe_attention * 0.5) * gpr_rank
            except RuntimeError as e:
                attention_cpu = attention.to('cpu')
                pe_attention_cpu = pe_attention.to('cpu')
                gpr_rank = gpr_rank.to('cpu')
                attention_cpu = attention_cpu * 0.5 + pe_attention_cpu * 0.5
                attention_cpu = attention_cpu * gpr_rank
                attention = attention_cpu.to('cuda')
        else:
            attention = attention * gpr_rank

        markoff_value = 0

        if ctx_mask is not None:
            attention = attention.masked_fill_(1 - ctx_mask.byte().unsqueeze(1), markoff_value)
            attention = attention.masked_fill_(1 - ctx_mask.byte().unsqueeze(-1), markoff_value)

        if self.epsilon is not None:
            if not self.epsilon == 0:
                attention = self.build_epsilon_neighbourhood(attention, self.epsilon, markoff_value)

        if self.topk is not None:
            attention = self.build_knn_neighbourhood(attention, self.topk, markoff_value)

        return attention


    def build_knn_neighbourhood(self, attention, topk, markoff_value):
        topk = min(topk, attention.size(-1))
        knn_val, knn_ind = torch.topk(attention, topk, dim=-1)
        weighted_adjacency_matrix = to_cuda((markoff_value * torch.ones_like(attention)).scatter_(-1, knn_ind, knn_val), self.device)
        return weighted_adjacency_matrix


    def build_epsilon_neighbourhood(self, attention, epsilon, markoff_value):
        mask = (attention > epsilon).detach().float()

        try:
            weighted_adjacency_matrix = attention * mask + markoff_value * (1 - mask)
        except:
            attention_np = attention.cpu().detach().numpy()
            mask_np = mask.cpu().detach().numpy()
            weighted_adjacency_matrix_np = attention_np * mask_np + markoff_value * (1 - mask_np)
            weighted_adjacency_matrix = torch.from_numpy(weighted_adjacency_matrix_np).to(self.device)

        return weighted_adjacency_matrix
    
def to_cuda(x, device=None):
    if device:
        x = x.to(device)
    return x