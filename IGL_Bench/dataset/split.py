import random
import torch
from torch_geometric.utils import degree
import numpy as np
import os

def get_classnum_imb_split(target_data, imb_level, shuffle_seed):
    random.seed(shuffle_seed)
    total_items = target_data.y.shape[0]
    shuffled_indices = list(range(total_items))
    random.shuffle(shuffled_indices)
    
    if imb_level == 'low':
        imb_ratio = 1
    elif imb_level == 'mid':
        imb_ratio = 20
    elif imb_level == 'high':
        imb_ratio = 100
        
    class_num_list, indices, inv_indices = sort(data=target_data)
    
    total_items = target_data.y.shape[0]
    n_classes = target_data.y.max().item() + 1
    n_train = int(total_items * 0.1)
    n_val = int(total_items * 0.1)
    n_test = total_items - n_train - n_val
    
    class_num_list_train = split_lt(
        class_num_list=class_num_list,
        indices=indices,
        inv_indices=inv_indices,
        imb_ratio=imb_ratio,
        n_cls=n_classes,
        n=n_train
    )

    n_val_per_class = n_val // n_classes
    class_num_list_val = torch.full((n_classes,), n_val_per_class, dtype=torch.long)
    remainder = n_val % n_classes
    if remainder > 0:
        class_num_list_val[:remainder] += 1

    shuffled_indices = torch.tensor(shuffled_indices, dtype=torch.long)

    train_indices = []
    val_indices = []
    test_indices = []

    for cls in range(n_classes):
        class_mask = (target_data.y[shuffled_indices] == cls)
        class_indices = shuffled_indices[class_mask]
        num_samples_in_class = len(class_indices)
        n_train_samples = class_num_list_train[cls].item()
        n_train_samples = min(n_train_samples, num_samples_in_class)
        train_class_indices = class_indices[:n_train_samples]
        train_indices.extend(train_class_indices.tolist())

        remaining_class_indices = class_indices[n_train_samples:]
        num_remaining = len(remaining_class_indices)

        n_val_samples = class_num_list_val[cls].item()
        n_val_samples = min(n_val_samples, num_remaining)
        val_class_indices = remaining_class_indices[:n_val_samples]
        val_indices.extend(val_class_indices.tolist())

        test_class_indices = remaining_class_indices[n_val_samples:]
        test_indices.extend(test_class_indices.tolist())

    train_mask = torch.zeros(target_data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(target_data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(target_data.num_nodes, dtype=torch.bool)

    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    target_data.train_mask = train_mask
    target_data.val_mask = val_mask
    target_data.test_mask = test_mask

    target_data.train_index = train_indices
    target_data.val_index = val_indices
    target_data.test_index = test_indices

    return target_data  

def split_nodes_by_degree(data, imb_level='low', seed=42):
    torch.manual_seed(seed)

    deg = degree(data.edge_index[0], num_nodes=data.num_nodes)
    deg_threshold = torch.quantile(deg.float(), 0.8)  
    high_deg_mask = deg >= deg_threshold  

    num_nodes = data.num_nodes
    num_classes = int(data.y.max().item()) + 1  

    total_train_size = num_nodes // 10  
    total_val_size   = num_nodes // 10  

    train_per_class = total_train_size // num_classes
    val_per_class   = total_val_size   // num_classes

    if imb_level == 'low':
        ratio = 0.1
    elif imb_level == 'mid':
        ratio = 0.2
    elif imb_level == 'high':
        ratio = 0.3

    train_index_list = []
    val_index_list   = []
    test_index_list  = []

    assigned_mask = torch.zeros(num_nodes, dtype=torch.bool)

    for c in range(num_classes):
        class_nodes = torch.where(data.y == c)[0]

        class_high_deg_nodes = class_nodes[high_deg_mask[class_nodes]]
        class_low_deg_nodes  = class_nodes[~high_deg_mask[class_nodes]]

        needed_train = train_per_class
        num_high_deg_needed = int(ratio * train_per_class)

        num_high_deg_for_train = min(num_high_deg_needed, needed_train)

        high_deg_train_nodes = class_high_deg_nodes[
            torch.randperm(len(class_high_deg_nodes))[:num_high_deg_for_train]
        ]

        remaining_for_train = needed_train - num_high_deg_for_train

        low_deg_train_nodes = class_low_deg_nodes[
            torch.randperm(len(class_low_deg_nodes))[:remaining_for_train]
        ]

        class_train_nodes = torch.cat([high_deg_train_nodes, low_deg_train_nodes], dim=0)
        train_index_list.append(class_train_nodes)
        assigned_mask[class_train_nodes] = True

        needed_val = val_per_class
        remain_nodes = class_nodes[~assigned_mask[class_nodes]]
        val_nodes = remain_nodes[torch.randperm(len(remain_nodes))[:needed_val]]
        val_index_list.append(val_nodes)
        assigned_mask[val_nodes] = True

        remain_nodes_after_val = class_nodes[~assigned_mask[class_nodes]]
        test_nodes = remain_nodes_after_val
        test_index_list.append(test_nodes)
        assigned_mask[test_nodes] = True

    train_index_tensor = torch.cat(train_index_list, dim=0)
    val_index_tensor   = torch.cat(val_index_list,   dim=0)
    test_index_tensor  = torch.cat(test_index_list,  dim=0)

    data.train_index = train_index_tensor.cpu().numpy()
    data.val_index   = val_index_tensor.cpu().numpy()
    data.test_index  = test_index_tensor.cpu().numpy()

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_index_tensor] = True

    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask[val_index_tensor] = True

    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[test_index_tensor] = True

    data.train_mask = train_mask
    data.val_mask   = val_mask
    data.test_mask  = test_mask

    return data

def get_topo_split_arxiv(target_data, shuffle_seed):
    all_idx = [i for i in range(target_data.num_nodes)]
    all_label = target_data.y.numpy()
    nclass = target_data.y.numpy().max().item() + 1
    random.seed(shuffle_seed)
    random.shuffle(all_idx) 
    
    num_samples = len(all_idx)
    train_each = int(num_samples * 0.1 / nclass)  
    valid_each = int(num_samples * 0.1 / nclass)  

    all_label = np.squeeze(all_label)
    label_counts = np.bincount(all_label, minlength=nclass) 

    train_list = [0] * nclass
    train_node = [[] for _ in range(nclass)]
    train_idx = []
    
    valid_list = [0] * nclass
    valid_idx = []

    after_train_idx = []

    train_class = [min(train_each, int(count * 0.1)) for count in label_counts]
    valid_class = [min(valid_each, int(count * 0.1)) for count in label_counts]

    for idx in all_idx:
        label = all_label[idx]
        if train_list[label] < train_class[label]:
            train_list[label] += 1
            train_node[label].append(idx)
            train_idx.append(idx)
        else:
            after_train_idx.append(idx)

    for idx in after_train_idx:
        label = all_label[idx]
        if valid_list[label] < valid_class[label]:
            valid_list[label] += 1
            valid_idx.append(idx)
    
    test_idx = list(set(after_train_idx) - set(valid_idx))
    
    train_mask = torch.zeros(target_data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(target_data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(target_data.num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[valid_idx] = True
    test_mask[test_idx] = True

    labeled_node = [[] for _ in range(nclass)]
    for idx in train_idx:
        label = all_label[idx]
        labeled_node[label].append(idx)

    target_data.train_index = train_idx
    target_data.val_index  = valid_idx
    target_data.test_index = test_idx

    target_data.train_mask = train_mask
    target_data.val_mask = val_mask
    target_data.test_mask = test_mask

    target_data.labeled_nodes = labeled_node

    return target_data
    

def get_topo_imb_split(target_data,shuffle_seed):
    all_idx = [i for i in range(target_data.num_nodes)]
    all_label = target_data.y.numpy()
    nclass = target_data.y.numpy().max().item() + 1
    random.seed(shuffle_seed)
    random.shuffle(all_idx) 
    
    train_each = (len(all_idx)*0.1) // nclass
    valid_each = (len(all_idx)*0.1) // nclass
    train_list = [0 for _ in range(nclass)]
    train_node = [[] for _ in range(nclass)]
    train_idx  = []
    
    for iter1 in all_idx:
        iter_label = all_label[iter1]
        if train_list[iter_label] < train_each:
            train_list[iter_label]+=1
            train_node[iter_label].append(iter1)
            train_idx.append(iter1)

        if sum(train_list)==train_each*nclass:break
    assert sum(train_list)==train_each*nclass
    
    after_train_idx = list(set(all_idx)-set(train_idx))
    random.shuffle(after_train_idx)
    
    valid_list = [0 for _ in range(nclass)]
    valid_idx  = []
    for iter2 in after_train_idx:
        iter_label = all_label[iter2]
        if valid_list[iter_label] < valid_each:
            valid_list[iter_label]+=1
            valid_idx.append(iter2)
        if sum(valid_list)==valid_each*nclass:break

    assert sum(valid_list)==valid_each*nclass
    test_idx = list(set(after_train_idx)-set(valid_idx))
    
    train_mask = torch.zeros(target_data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(target_data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(target_data.num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[valid_idx] = True
    test_mask[test_idx] = True

    labeled_node = [[] for _ in range(nclass)]
    for idx in train_idx:
        label = all_label[idx]
        labeled_node[label].append(idx)

    target_data.train_index = train_idx
    target_data.val_index  = valid_idx
    target_data.test_index = test_idx

    target_data.train_mask = train_mask
    target_data.val_mask = val_mask
    target_data.test_mask = test_mask

    target_data.labeled_nodes = labeled_node

    return target_data

def load_split(data_name,imb_level):
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'graph_topology_imbalance', data_name)
    load_file = os.path.join(path, 'split_' + imb_level + '.pt')
    loaded_split = torch.load(load_file, map_location=torch.device('cpu'))
    train_mask = loaded_split['train_mask']
    val_mask = loaded_split['val_mask']
    test_mask = loaded_split['test_mask']
    
    return train_mask, val_mask, test_mask    

def get_shuffle_seed(data_name, imb_level):
    seed_mapping = {
        'Cora': {'low': 25, 'high': 38},
        'CiteSeer': {'low': 34, 'high': 47},
        'PubMed': {'low': 44, 'high': 48},
        'Chameleon': {'low': 5, 'high': 14},
        'Squirrel': {'low': 100, 'high': 50},
        'Actor': {'low': 40, 'high': 5},
        'Photo': {'low': 13, 'high': 24},
        'Computers': {'low': 1, 'high': 100},
        'ogbn-arxiv': {'low': 1, 'high': 11}
    }

    if imb_level not in ['low', 'high']:
        raise ValueError(f"invalid imbalance level: '{imb_level}'. only for 'low' or 'high'.")

    return seed_mapping[data_name][imb_level]

def sort(data, data_mask=None):
    if data_mask is None:
        y = data.y
    else:
        y = data.y[data_mask]
    
    n_cls = data.y.max().item() + 1

    class_num_list = []
    for i in range(n_cls):
        class_num_list.append(int((y == i).sum().item()))
    
    class_num_list_tensor = torch.tensor(class_num_list)
    class_num_list_sorted_tensor, indices = torch.sort(class_num_list_tensor, descending=True)
    inv_indices = torch.zeros(n_cls, dtype=indices.dtype, device=indices.device)
    for i in range(n_cls):
        inv_indices[indices[i].item()] = i

    assert torch.equal(class_num_list_sorted_tensor, class_num_list_tensor[indices])
    assert torch.equal(class_num_list_tensor, class_num_list_sorted_tensor[inv_indices])
    return class_num_list_tensor, indices, inv_indices

def split_lt(class_num_list, indices, inv_indices, imb_ratio, n_cls, n, keep=0):
    class_num_list = class_num_list[indices]  # sort
    mu = np.power(imb_ratio, 1 / (n_cls - 1))
    _mu = 1 / mu
    if imb_ratio == 1:
        n_max = n / n_cls
    else:
        n_max = n / (imb_ratio * mu - 1) * (mu - 1) * imb_ratio
    class_num_list_lt = []
    for i in range(n_cls):
        class_num_list_lt.append(round(min(max(n_max * np.power(_mu, i), 1), class_num_list[i].item() - keep)))
    class_num_list_lt = torch.tensor(class_num_list_lt)
    return class_num_list_lt[inv_indices]  # unsort