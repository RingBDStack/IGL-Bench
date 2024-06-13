import torch
import numpy as np
from torch_scatter import scatter_add


def get_dataset(name, path, split_type='public'):
    import torch_geometric.transforms as T

    if name == "Cora" or name == "CiteSeer" or name == "PubMed":
        from torch_geometric.datasets import Planetoid
        dataset = Planetoid(path, name, transform=T.NormalizeFeatures(), split=split_type)
    elif name == "chameleon" or name == "squirrel":
        from dataset.WikipediaNetwork import WikipediaNetwork
        dataset = WikipediaNetwork(path, name, transform = T.NormalizeFeatures())
    elif name == "Wisconsin":
        # from torch_geometric.datasets import  WebKB
        from dataset.WebKB import WebKB
        dataset = WebKB(path, name, transform = T.NormalizeFeatures())
    elif name == 'actor':
        from torch_geometric.datasets import Actor
        dataset = Actor(path)
    elif name == 'arxiv':
        from ogb.nodeproppred import PygNodePropPredDataset
        dataset = PygNodePropPredDataset(name='ogbn-arxiv',root=path)
    elif name == 'photo' or name == 'computers':
        from torch_geometric.datasets import Amazon
        dataset = Amazon(path, name=name) 
    else:
        raise NotImplementedError("Not Implemented Dataset!")

    return dataset


def split_semi_dataset(total_node, n_data, n_cls, class_num_list, idx_info, device):
    new_idx_info = []
    _train_mask = idx_info[0].new_zeros(total_node, dtype=torch.bool, device=device)
    for i in range(n_cls):
        if n_data[i] > class_num_list[i]:
            cls_idx = torch.randperm(len(idx_info[i]))
            cls_idx = idx_info[i][cls_idx]
            cls_idx = cls_idx[:class_num_list[i]]
            new_idx_info.append(cls_idx)
        else:
            new_idx_info.append(idx_info[i])
        _train_mask[new_idx_info[i]] = True

    assert _train_mask.sum().long() == sum(class_num_list)
    assert sum([len(idx) for idx in new_idx_info]) == sum(class_num_list)

    return _train_mask, new_idx_info


def get_idx_info(label, n_cls, train_mask):
    index_list = torch.arange(len(label)).to('cuda')
    idx_info = []
    for i in range(n_cls):
        cls_indices = index_list[((label == i) & train_mask)]
        idx_info.append(cls_indices)
    return idx_info

def get_idx_info_arxiv(label, n_cls, train_mask):
    label = np.squeeze(label)
    label = label.cpu()
    train_mask = train_mask.cpu()

    label = label.to(torch.int32)
    train_mask = train_mask.to(torch.bool)

    idx_info = []
    
    batch_size = 10000 
    num_batches = (len(label) + batch_size - 1) // batch_size

    for i in range(n_cls):
        cls_indices = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(label))
            
            batch_labels = label[start_idx:end_idx]
            batch_mask = train_mask[start_idx:end_idx]
            batch_index_list = torch.arange(start_idx, end_idx, dtype=torch.int32)
            
            mask = (batch_labels == i) & batch_mask
            cls_indices_batch = batch_index_list[mask]
            cls_indices.append(cls_indices_batch)

        cls_indices = torch.cat(cls_indices).to('cuda')
        idx_info.append(cls_indices)

    return idx_info
