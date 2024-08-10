import torch
import pickle
import logging
import sys
import dgl
import torch.nn.functional as F 
from tqdm import tqdm 
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

def evaluate_gnn(g, features, labels, mask, model, save_res_dir='none', dataset=None):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits_ = logits
    if dataset == 'yelp':
        logits_ = (torch.sigmoid(logits_) > 0.5)
        metrics = f1_score(labels[mask].cpu().numpy(), logits_[mask].cpu().numpy(), average='micro')
    else:
        _, indices_ = torch.max(logits_, dim=1)
        logits = logits[mask]
        labels_ = labels
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        metrics = correct.item() * 1.0 / len(labels)
        if save_res_dir != 'none':
            pickle.dump([logits_.cpu(), indices_.cpu(), labels_.cpu()], open(save_res_dir, 'wb'))

    return metrics, f1_score(labels.cpu().numpy(), indices.cpu().numpy(), average='macro')
    
def accuracy(logits, label):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == label)
    return correct.item() * 1.0 / len(label)

def f1_macro(logits, label):
    _, indices = torch.max(logits, dim=1)
    return f1_score(label.cpu().numpy(), indices.cpu().numpy(), average='macro')

def bacc(logits, label):
    _, indices = torch.max(logits, dim=1)
    return balanced_accuracy_score(label.cpu().numpy(), indices.cpu().numpy())

def roc_auc(logits, label):
    # _, indices = torch.max(logits, dim=1)
    one_hot_labels = label_binarize(label, classes=list(range(logits.size(1))))
    return roc_auc_score(one_hot_labels, logits.cpu().numpy(), multi_class='ovo')

def init_logger(filename=None):
    logger = logging.getLogger(__name__)
    handlers = [logging.StreamHandler(sys.stdout)]
    if filename is not None:
        handlers.append(logging.FileHandler(filename=filename))
    logging.basicConfig(
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        handlers=handlers,
    )

    return logger

def save_model(state_dict, gnn_config, dir):
    pickle.dump(gnn_config, open('{}.config'.format(dir), 'wb'))
    torch.save({k:v.cpu() for k, v in state_dict.items()}, '{}.ckpt'.format(dir))

def filter_glist(glist, info_dict, degree_train):
    new_glist = []
    new_inverse_indices = []
    new_degrees = []
    for i, degree in enumerate(info_dict['degrees']):
        if degree >= degree_train:
            new_glist.append(glist[i])
            new_inverse_indices.append(info_dict['inverse_indices'][i])
            new_degrees.append(degree)
    return new_glist, {'inverse_indices': torch.tensor(new_inverse_indices), 'degrees':torch.tensor(new_degrees)}

def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss

def kl_div(x, y):
    x = F.log_softmax(x, dim=1)
    y = F.softmax(y, dim=1)
    return F.kl_div(x, y, reduction='batchmean')

def memax(x, y, normalizer=None):
    if normalizer == None:
        x = F.softmax(x, dim=1)
        y = F.softmax(y, dim=1)
        return F.cross_entropy(x, y)
    normalizer = torch.clamp(normalizer, min=0.3, max=1)
    normalizer = normalizer.reshape(-1, 1)
    x = F.softmax(x/normalizer, dim=1)
    y = F.softmax(y/normalizer, dim=1)
    return F.cross_entropy(x, y)

def memax_2(x, y):
    x = F.softmax(x, dim=1)
    y = F.softmax(y, dim=1)
    # x = F.softmax(x, dim=1)
    # y = F.softmax(y, dim=1)
    return kl_div(x.t(), y.t())

def divide_chunks(l, n):
    # looping till length l
    for i, j in enumerate(range(0, len(l), n)):
        yield i*n, i*n+len(l[j:j + n]), l[j:j + n]

def inject_nodes(batched_masked_graphs, generated_neighbors, masked_offset, device, mask=None):
    assert len(masked_offset) == len(generated_neighbors)
    batched_masked_graphs_ = dgl.add_nodes(batched_masked_graphs, len(masked_offset), {'feat':generated_neighbors})
    temp = torch.arange(batched_masked_graphs_.number_of_nodes() - len(masked_offset), batched_masked_graphs_.number_of_nodes()).to(device)
    masked_offset = masked_offset.to(device)
    # src = torch.cat([temp, masked_offset])
    # dst = torch.cat([masked_offset, temp])
    src = temp[mask] if mask != None else temp 
    dst = masked_offset[mask] if mask != None else masked_offset
    batched_masked_graphs_.add_edges(src, dst)
    return batched_masked_graphs_

def evaluate_generator(dataloader, generator, GNN, label, device, args, return_dist=False, iteration=1, upperbound=0):
    if upperbound == 0:
        upperbound = 1e5
        
    recon_dist_list = {}
    for i in range(iteration): recon_dist_list[i] = []
    generator.eval()
    for _, data in enumerate(tqdm(dataloader) if args.bar else dataloader):
        batched_graphs, offset  = data
        batched_graphs = batched_graphs.to(device)
        for itera in range(iteration):
            with torch.no_grad():
                generated_neighbors = generator(batched_graphs, offset)
            batched_graphs = inject_nodes(batched_graphs, generated_neighbors, offset, device)
            with torch.no_grad():
                recon_dist_list[itera].append(GNN(batched_graphs, batched_graphs.ndata['feat'])[offset].cpu())
    returns = []
    for i in range(iteration):
        recon_dist = torch.cat(recon_dist_list[i])
        current_acc = accuracy(recon_dist, label)
        current_f1 = f1_macro(recon_dist, label)
        entropy = F.cross_entropy(recon_dist, label.long())
        current_bacc = bacc(recon_dist, label)
        current_roc_auc = roc_auc(recon_dist, label)

        if return_dist:
            returns.append([current_acc, entropy.item(), recon_dist, current_f1, current_bacc, current_roc_auc])
        else:
            returns.append([current_acc, entropy.item(), current_f1, current_bacc, current_roc_auc])
    
    return returns 