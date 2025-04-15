import torch
from torch.utils.data import Sampler
from torch_geometric.data import Dataset

class MyDataset(Dataset):
    def __init__(self, full_dataset, sampled_list):
        self.full_dataset = full_dataset
        self.sampled_list = sampled_list

    def __len__(self):
        return len(self.sampled_list)

    def __getitem__(self, idx):
        actual_idx = self.sampled_list[idx]
        data = self.full_dataset[actual_idx] 
        return data, actual_idx  

class IndexSampler(Sampler):
    def __init__(self, sampled_list):
        self.sampled_list = sampled_list

    def __iter__(self):
        return iter(self.sampled_list)

    def __len__(self):
        return len(self.sampled_list)

