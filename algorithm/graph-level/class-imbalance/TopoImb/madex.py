import os
import numpy as np
import torch
import torch.nn.functional as F

from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree, to_networkx
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
import dill as pickle 


class MyOneHotDegree(BaseTransform):
    r"""Adds the node degree as one hot encodings to the node features
    (functional name: :obj:`one_hot_degree`).

    Args:
        max_degree (int): Maximum degree.
        in_degree (bool, optional): If set to :obj:`True`, will compute the
            in-degree of nodes instead of the out-degree.
            (default: :obj:`False`)
        cat (bool, optional): Concat node degrees to node features instead
            of replacing them. (default: :obj:`True`)
    """

    def __init__(
            self,
            max_degree: int,
            in_degree: bool = False,
            cat: bool = False,
            unique_set: set = None,
    ):
        self.max_degree = max_degree
        self.in_degree = in_degree
        self.cat = cat
        self.unique_set = sorted(list(unique_set))

    def __call__(self, data: Data) -> Data:
        idx= data.edge_index[1 if self.in_degree else 0]
        deg = degree(idx, data.num_nodes, dtype=torch.long)
        index = np.searchsorted(self.unique_set, deg)
        tensor_index = torch.tensor(index, dtype=torch.long)
        #self.max_degree=tensor_index.max().item() + 1
        deg = F.one_hot(tensor_index, num_classes=self.max_degree+1).to(torch.float)
        data.x = deg

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.max_degree})'


def get_unique_degrees(dataset):
    max_degree = 0
    unique_degrees = set()

    for data in dataset:
        G = to_networkx(data)
        degrees = np.array([G.degree[n] for n in G.nodes])
        unique_degrees.update(np.unique(degrees).tolist())
        if degrees.size == 0:
            temp = 0
        else:
            temp = np.max(degrees)
        if temp > max_degree:
            max_degree = np.max(degrees)
    return max_degree, unique_degrees
