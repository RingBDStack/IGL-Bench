from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, balanced_accuracy_score
import torch
import torch.nn.functional as F

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def Roc_F(logits, labels, pre='valid'):
    if labels.max() > 1:#require set(labels) to be the same as columns of logits 
        auc_score = roc_auc_score(labels.detach().cpu(), F.softmax(logits, dim=-1).detach().cpu(), average='macro', multi_class='ovr')
    else:
        auc_score = roc_auc_score(labels.detach().cpu(), F.softmax(logits, dim=-1)[:,1].detach().cpu(), average='macro')

    macro_F = f1_score(labels.detach().cpu(), torch.argmax(logits, dim=-1).detach().cpu(), average='macro')

    return auc_score, macro_F

def compute_metrics(logits, labels):
    preds = torch.argmax(logits, dim=-1)

    acc = accuracy_score(labels.detach().cpu(), preds.detach().cpu())

    bacc = balanced_accuracy_score(labels.detach().cpu(), preds.detach().cpu())

    mf1 = f1_score(labels.detach().cpu(), preds.detach().cpu(), average='macro')

    if labels.max() > 1:  
        auc_roc = roc_auc_score(labels.detach().cpu(), F.softmax(logits, dim=-1).detach().cpu(), average='macro', multi_class='ovr')
    else:  
        auc_roc = roc_auc_score(labels.detach().cpu(), F.softmax(logits, dim=-1)[:, 1].detach().cpu(), average='macro')

    return acc, bacc, mf1, auc_roc