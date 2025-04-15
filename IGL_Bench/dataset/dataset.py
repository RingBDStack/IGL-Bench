from enum import Enum
import os
from .load_graph import TUDataset, OGBDataset, get_class_num, shuffle
from .load_node import load_node_data
from ogb.graphproppred import PygGraphPropPredDataset
import torch_geometric.transforms as T
from collections import Counter
import numpy as np
import torch
import random

from .split import get_shuffle_seed, get_topo_imb_split, split_nodes_by_degree, get_classnum_imb_split, load_split
from .split import  get_topo_split_arxiv

name_mapping = {
    "PTC-MR": "PTC_MR",
    "FRANKENSTEIN": "FRANKENSTEIN",
    "PROTEINS": "PROTEINS",
    "IMDB-B": "IMDB-BINARY",
    "REDDIT-B": "REDDIT-BINARY",
    "COLLAB": "COLLAB",  
    "D&D": "DD",
    "ogbg-molhiv": "ogbg-molhiv",         
}

graph_imb_level_mapping = {
    'low': 0.5,
    'mid': 0.7,
    'high': 0.9
}

class ImbalanceLevel(Enum):
    low = "low"
    mid = "mid"
    high = "high"

class Dataset:
    def __init__(self, task, data_name, imb_type, imb_level, data_path='data', 
                 pre_transform=T.ToSparseTensor(remove_edge_index=False), shuffle_seed=10):
        if task not in ["node", "graph"]:
            raise ValueError("Invalid task type. Must be 'node' or 'graph'.")
        
        if task == "node":
            valid_node_data_names = ["Cora", "CiteSeer", "PubMed", "Photo", "Computers", "ogbn-arxiv", "Chameleon", "Squirrel", "Actor"]
            if data_name not in valid_node_data_names:
                raise ValueError(f"Invalid data_name for 'node' task. Must be one of {valid_node_data_names}.")
            
        elif task == "graph":
            valid_graph_data_names = ["PTC-MR", "FRANKENSTEIN", "PROTEINS", "IMDB-B", "REDDIT-B", "ogbg-molhiv", "COLLAB", "D&D"]
            if data_name not in valid_graph_data_names:
                raise ValueError(f"Invalid data_name for 'graph' task. Must be one of {valid_graph_data_names}.")        
        
        if task == "node":
            valid_node_imb_types = ["class", "topo_local", "topo_global"]
            if imb_type not in valid_node_imb_types:
                raise ValueError(f"Invalid imb_type for 'node' task. Must be one of {valid_node_imb_types}.")
        
        elif task == "graph":
            valid_graph_imb_types = ["class", "topology"]
            if imb_type not in valid_graph_imb_types:
                raise ValueError(f"Invalid imb_type for 'graph' task. Must be one of {valid_graph_imb_types}.")
            
        if imb_level in ImbalanceLevel.__members__:
            self.imb_level = imb_level 
        else:
            raise ValueError(f"Invalid imbalance level. Must be one of {list(ImbalanceLevel)}.")
        
        if task == "graph":
            data_name = name_mapping[data_name]
                
        self.task = task 
        self.data_name = data_name
        self.imb_type = imb_type
        self.imb_level = imb_level
        self.data_path = data_path
        self.pre_transform = pre_transform
        
        if task == 'graph':
            dataset = self.load_graph_data(data_name, data_path, pre_transform)
            for data in dataset:
                data.x = data.x.float()
                if data.y.dim() > 1:
                    data.y = data.y.view(-1)
            dataset.y = torch.tensor([data.y.item() for data in dataset],dtype=torch.int32)
            
            if self.imb_type == 'class':
                num_train = (int)(len(dataset) * 0.1)
                num_val = (int)(len(dataset) * 0.1 / dataset.num_classes)
                imb_ratio = graph_imb_level_mapping.get(self.imb_level)
                labels = [data.y.item() for data in dataset]
                n_data = Counter(labels)
                n_data = np.array(list(n_data.values()))
                c_train_num, c_val_num = get_class_num(imb_ratio, num_train, num_val, self.data_name,
                                                    dataset.num_classes,n_data)
            
                
                dataset.train_index, dataset.val_index, dataset.test_index = shuffle(
                        dataset, c_train_num, c_val_num, dataset.y)
                
                train_mask = torch.zeros(dataset.y.size(0), dtype=torch.bool)
                val_mask = torch.zeros(dataset.y.size(0), dtype=torch.bool)
                test_mask = torch.zeros(dataset.y.size(0), dtype=torch.bool)

                train_mask[dataset.train_index] = True
                val_mask[dataset.val_index] = True
                test_mask[dataset.test_index] = True

                dataset.train_mask = train_mask
                dataset.val_mask = val_mask
                dataset.test_mask = test_mask  
            
            elif imb_type == 'topology':
                train_mask, val_mask, test_mask = load_split(data_name,imb_level)
                dataset.train_mask = train_mask
                dataset.val_mask = val_mask
                dataset.test_mask = test_mask
                dataset.train_index = torch.nonzero(train_mask, as_tuple=False).squeeze().numpy()
                dataset.val_index = torch.nonzero(val_mask, as_tuple=False).squeeze().numpy()
                dataset.test_index = torch.nonzero(test_mask, as_tuple=False).squeeze().numpy()
                              
        elif task == 'node':
            dataset = load_node_data(data_name, data_path)
            dataset.data_name = data_name
            
            if imb_type == 'topo_global':
                shuffle_seed = get_shuffle_seed(data_name,imb_level)
                if data_name == 'ogbn-arxiv':
                    datatset = get_topo_split_arxiv(dataset,shuffle_seed)
                else:
                    dataset = get_topo_imb_split(dataset,shuffle_seed)
                
            elif imb_type == 'topo_local':
                dataset = split_nodes_by_degree(dataset,imb_level,shuffle_seed)
            
            elif imb_type == 'class':
                dataset = get_classnum_imb_split(dataset,imb_level,shuffle_seed)
                        
        self.dataset = dataset  
    
    def load_dataset(self):
        return self.dataset
        
    def load_graph_data(self, data_name, data_path, pre_transform):
        if data_name in ["PTC_MR", "FRANKENSTEIN", "PROTEINS", "IMDB-BINARY", "REDDIT-BINARY",  "COLLAB", "DD"]:
            return get_TUDataset(data_name, data_path, pre_transform)
        else:
            return get_OGBataset(data_name, data_path, pre_transform) 

        
def get_TUDataset(dataset, data_path, pre_transform):
    """
    'PROTEINS', 'REDDIT-BINARY', 'MUTAG', 'PTC_MR', 'DD', 'NCI1'
    """
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..',data_path, 'TU')
    dataset = TUDataset(path, name=dataset, pre_transform=pre_transform)

    return dataset

def get_OGBataset(dataset, data_path, pre_transform=T.ToSparseTensor()):
    """
    'ogbg-molhiv', 'ogbg-molpcba'
    """
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', data_path, 'OGB')
    dataset = PygGraphPropPredDataset(root=path,name=dataset,pre_transform=pre_transform)

    return dataset