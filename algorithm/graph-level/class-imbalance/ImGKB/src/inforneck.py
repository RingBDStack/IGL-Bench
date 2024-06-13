import torch
import torch.nn.functional as F
import torch.nn as nn
def MI_Est(discriminator, embeddings, positive):
    eps = 1e-10
    batch_size = embeddings.shape[0]
    shuffle_embeddings = positive[torch.randperm(batch_size)]
    joint = discriminator(embeddings,positive)
    margin = discriminator(embeddings, shuffle_embeddings)
    joint = joint + eps
    margin = margin + eps
    swich = 'Donsker'
    if swich=='Donsker':
        mi_est = torch.mean(joint) + torch.clamp(torch.log(torch.mean(torch.exp(margin))),-10000,10000)
    elif swich=='JSD':
        mi_est = -torch.mean(F.softplus(-joint)) - torch.mean(F.softplus(-margin)+margin)
    elif swich=='x^2':
        mi_est = torch.mean(joint**2) - 0.5* torch.mean((torch.sqrt(margin**2)+1.0)**2)
    return mi_est

class InBo(torch.nn.Module):
    def __init__(self, hidden_size):
        super(InBo, self).__init__()

        self.input_size = hidden_size
        self.hidden_size = hidden_size
        self.lin1 = torch.nn.Linear(self.input_size,self.hidden_size)
        self.lin2 = torch.nn.Linear(self.hidden_size, 1)
        self.relu = torch.nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
    def forward(self, embeddings,positive):
        cat_embeddings = torch.cat((embeddings, positive),dim = 1)
        pre = self.relu(self.lin1(cat_embeddings))
        pre = self.relu(self.lin2(pre))
        return pre

class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1, bias=False)
        )
    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta