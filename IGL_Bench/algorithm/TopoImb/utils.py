from sklearn.metrics import roc_auc_score, f1_score
import torch
import torch.nn.functional as F

def accuracy(logits, labels):
    preds = logits.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def grouped_accuracy(logits, labels, group_labels):
    preds = logits.argmax(1)
    group_ac_dict={}
    for group in set(group_labels):
        group_idx = group_labels==group
        group_ac = (preds[group_idx]==labels[group_idx]).sum()/(group_idx.sum()+0.00000001)
        group_ac_dict[group] = group_ac

    return group_ac_dict

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
    if labels.max() > 1:#require set(labels) to be the same as columns of logits 
        auc_score = roc_auc_score(labels.detach().cpu(), F.softmax(logits, dim=-1).detach().cpu(), average='macro', multi_class='ovr')
    else:
        auc_score = roc_auc_score(labels.detach().cpu(), F.softmax(logits, dim=-1)[:,1].detach().cpu(), average='macro')

    macro_F = f1_score(labels.detach().cpu(), torch.argmax(logits, dim=-1).detach().cpu(), average='macro')

    return auc_score, macro_F

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