import torch
from sklearn.metrics import f1_score
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score,accuracy_score, roc_auc_score
from torch.nn.functional import softmax


'''
def test(model, data, train_mask, val_mask, test_mask, alpha):
    with torch.no_grad():
        model.eval()
        logits, accs = model(data, alpha), []
        for mask in [train_mask,val_mask,test_mask]:
            pred = logits[mask].max(1)[1]
            acc = f1_score(pred.cpu(), data.y[mask].cpu(), average='micro')
            accs.append(acc)

        accs.append(F.nll_loss(model(data, alpha)[val_mask], data.y[val_mask]))
        accs.append(f1_score(pred.cpu(), data.y[mask].cpu(), average='weighted'))
    return accs
'''

def test(model, data, train_mask, val_mask, test_mask, alpha):
    model.eval()
    with torch.no_grad():
        logits = model(data, alpha)
        probs = softmax(logits, dim=1) 
        accs = []

        for mask in [train_mask, val_mask, test_mask]:
            pred = logits[mask].max(1)[1]
            acc = f1_score(data.y[mask].cpu(), pred.cpu(), average='micro')
            accs.append(acc)

        accs.append(torch.nn.functional.nll_loss(torch.log(probs[val_mask]), data.y[val_mask]))

        test_pred = logits[test_mask].max(1)[1]
        test_f1_weighted = f1_score(test_pred.cpu(), data.y[test_mask].cpu(), average='weighted')
        accs.append(test_f1_weighted)

        test_f1_macro = f1_score(test_pred.cpu(), data.y[test_mask].cpu(), average='macro')
        test_bal_acc = balanced_accuracy_score(data.y[test_mask].cpu().numpy(), test_pred.cpu().numpy())
        accs.extend([test_f1_macro, test_bal_acc])


        if data.y.max() > 1:  
            test_auroc = roc_auc_score(data.y[test_mask].cpu().numpy(), probs[test_mask].cpu().numpy(), multi_class='ovr')
        else:  
            test_auroc = roc_auc_score(data.y[test_mask].cpu().numpy(), probs[test_mask][:,1].cpu().numpy())

        accs.append(test_auroc)

    return accs
    