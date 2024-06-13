import torch_geometric.datasets as dt
import torch_geometric.transforms as T
import torch
import numpy as np
from dgl.data.utils import generate_mask_tensor, idx2mask
from sklearn.model_selection import train_test_split
import random

def select_dataset(ds,spcial,shuffle_seed):
    if ds=='Cora' or ds=='Citeseer' or ds=='pubmed':
        ds_loader='Planetoid'
    elif ds=='Photo' or ds == 'computers':
        ds_loader='Amazon'
    elif ds == 'chameleon' or ds == 'Squirrel':
        ds_loader='WikipediaNetwork'
    else:
        ds_loader=ds
    dataset=load_datas(ds_loader,ds,spcial,shuffle_seed)
    if ds == 'Actor':
        data=dataset.data
        dataset.name = ds
    else:
        data=dataset[0]

    train_mask=data.train_mask
    val_mask=data.val_mask
    test_mask=data.test_mask
    return dataset,data,train_mask,val_mask,test_mask

def load_datas(ds_loader,ds,spcial,shuffle_seed):
    if ds_loader=='Planetoid':
        dataset = dt.Planetoid(root='data/'+ds, name=ds, transform=T.NormalizeFeatures())
    else:
        dataset = getattr(dt, ds_loader)('data/'+ds,ds)

    if ds_loader == 'Actor':
        
        dataset.name = ds

    data = get_split(dataset, spcial,shuffle_seed)
    dataset.data = data
    return dataset

def get_split(dataset, spcial,shuffle_seed):
    data = dataset.data   
    all_label=data.y.numpy()
    nclass = np.max(data.y.numpy())+1
    #values=np.load('hyperemb/'+dataset.name+'_values.npy')
    #sorted, indices = torch.sort(torch.norm(torch.tensor(values),dim=1),descending=True)
    #train set split ratio 1:1:8
    if spcial == 1:#Top 50% in the Poincare weight
        train_idx, val_idx, test_idx = split_idx1(indices[:data.num_nodes//2],indices[data.num_nodes//2:], 0.2, 0.1, 42)
    elif spcial == 2:#Bottom 50%
        train_idx, val_idx, test_idx = split_idx1(indices[data.num_nodes//2:],indices[:data.num_nodes//2], 0.2, 0.1, 42)
    elif spcial == 3:#Top 33%
        train_idx, val_idx, test_idx = split_idx1(indices[:data.num_nodes//3],indices[data.num_nodes//3:], 0.3, 0.1, 42)
    elif spcial == 4:#Middle 33%
        remaining = torch.cat((indices[:data.num_nodes//3],indices[data.num_nodes//3+data.num_nodes//3:]))
        train_idx, val_idx, test_idx = split_idx1(indices[data.num_nodes//3:data.num_nodes//3+data.num_nodes//3],remaining, 0.3, 0.1, 42)
    elif spcial == 5:#Bottom 33%
        train_idx, val_idx, test_idx = split_idx1(indices[data.num_nodes//3+data.num_nodes//3:],indices[:data.num_nodes//3+data.num_nodes//3], 0.3, 0.1, 42)
    else:#random
        train_idx, val_idx, test_idx = split_idx(np.arange(data.num_nodes), all_label,nclass, shuffle_seed)

    data.train_mask = generate_mask_tensor(idx2mask(train_idx, data.num_nodes))
    data.val_mask = generate_mask_tensor(idx2mask(val_idx, data.num_nodes))
    data.test_mask = generate_mask_tensor(idx2mask(test_idx, data.num_nodes))
    return data

''' 
def split_idx(samples, train_size, val_size, random_state=None):
    train, val = train_test_split(samples, train_size=train_size, random_state=random_state)
    if isinstance(val_size, float):
        val_size *= len(samples) / len(val)
    val, test = train_test_split(val, train_size=val_size, random_state=random_state)
    return train, val, test
   
'''  
def split_idx(all_idx, all_label,nclass, shuffle_seed):
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

    return train_idx,valid_idx,test_idx

def split_idx1(samples1, samples2, train_size, val_size, random_state=None):
    train, val = train_test_split(samples1, train_size=train_size, random_state=random_state)
    val = torch.cat((val,samples2))
    val, test = train_test_split(val, train_size=val_size, random_state=random_state)
    return train, val, test
