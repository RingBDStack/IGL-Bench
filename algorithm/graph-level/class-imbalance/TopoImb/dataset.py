import os.path as osp
import torch

import torch
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.utils import degree
from torch_geometric.io import read_tu_data
import torch
from typing import Optional, Callable, List
from itertools import repeat, product
import torch.nn.functional as F
import numpy as np
import os
import shutil
import random
import dill as pickle


import os
import os.path as osp
import shutil
from typing import Callable, List, Optional

import torch

from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.io import read_tu_data
from madex import MyOneHotDegree,get_unique_degrees


class TUDataset(InMemoryDataset):
    r"""A variety of graph kernel benchmark datasets, *.e.g.* "IMDB-BINARY",
    "REDDIT-BINARY" or "PROTEINS", collected from the `TU Dortmund University
    <https://chrsmrrs.github.io/datasets>`_.
    In addition, this dataset wrapper provides `cleaned dataset versions
    <https://github.com/nd7141/graph_datasets>`_ as motivated by the
    `"Understanding Isomorphism Bias in Graph Data Sets"
    <https://arxiv.org/abs/1910.12091>`_ paper, containing only non-isomorphic
    graphs.

    .. note::
        Some datasets may not come with any node labels.
        You can then either make use of the argument :obj:`use_node_attr`
        to load additional continuous node attributes (if present) or provide
        synthetic node features using transforms such as
        like :class:`torch_geometric.transforms.Constant` or
        :class:`torch_geometric.transforms.OneHotDegree`.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The `name
            <https://chrsmrrs.github.io/datasets/docs/datasets/>`_ of the
            dataset.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        use_node_attr (bool, optional): If :obj:`True`, the dataset will
            contain additional continuous node attributes (if present).
            (default: :obj:`False`)
        use_edge_attr (bool, optional): If :obj:`True`, the dataset will
            contain additional continuous edge attributes (if present).
            (default: :obj:`False`)
        cleaned: (bool, optional): If :obj:`True`, the dataset will
            contain only non-isomorphic graphs. (default: :obj:`False`)
    """

    url = 'https://www.chrsmrrs.com/graphkerneldatasets'
    cleaned_url = ('https://raw.githubusercontent.com/nd7141/'
                   'graph_datasets/master/datasets')

    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 use_node_attr: bool = False, use_edge_attr: bool = False,
                 cleaned: bool = False):
        self.name = name
        self.cleaned = cleaned
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        if self.data.x is not None and not use_node_attr:
            num_node_attributes = self.num_node_attributes
            self.data.x = self.data.x[:, num_node_attributes:]
        if self.data.edge_attr is not None and not use_edge_attr:
            num_edge_attributes = self.num_edge_attributes
            self.data.edge_attr = self.data.edge_attr[:, num_edge_attributes:]
        '''
        if not (self.name == 'MUTAG' or self.name == 'PTC_MR' or self.name == 'DD' or self.name == 'PROTEINS' or self.name == 'NCI1' or self.name == 'NCI109' or self.name == 'DHFR'):
            edge_index = self.data.edge_index[0, :].numpy()
            _, num_edge = self.data.edge_index.size()
            nlist = [
                edge_index[n] + 1 for n in range(num_edge - 1) if edge_index[n] > edge_index[n + 1]]
            nlist.append(edge_index[-1] + 1)

            num_node = np.array(nlist).sum()
            self.data.x = torch.ones((num_node, 1))

            # deg = degree(self.data.edge_index[0], num_node, dtype = torch.long)
            # self.data.x = F.one_hot(deg).type(torch.float)

            edge_slice = [0]
            k = 0
            for n in nlist:
                k = k + n
                edge_slice.append(k)
            self.slices['x'] = torch.tensor(edge_slice)

        # print(self.data.edge_index)
        '''    
        if self.name in['REDDIT-MULTI-5K',"FRANKENSTEIN",'IMDB-BINARY','REDDIT-BINARY',"COLLAB","TRIANGLES"]:
            self.num_feature = self.data.x.size(1)
        else:
            self.num_feature = self.num_features
        self.data.id = torch.arange(0, self.data.y.size(0))
        self.slices['id'] = self.slices['y'].clone()

    @property
    def raw_dir(self) -> str:
        name = f'raw{"_cleaned" if self.cleaned else ""}'
        return osp.join(self.root, self.name, name)

    @property
    def processed_dir(self) -> str:
        name = f'processed{"_cleaned" if self.cleaned else ""}'
        return osp.join(self.root, self.name, name)

    @property
    def num_node_labels(self) -> int:
        if self.data.x is None:
            return 0
        for i in range(self.data.x.size(1)):
            x = self.data.x[:, i:]
            if ((x == 0) | (x == 1)).all() and (x.sum(dim=1) == 1).all():
                return self.data.x.size(1) - i
        return 0

    @property
    def num_node_attributes(self) -> int:
        if self.data.x is None:
            return 0
        return self.data.x.size(1) - self.num_node_labels

    @property
    def num_edge_labels(self) -> int:
        if self.data.edge_attr is None:
            return 0
        for i in range(self.data.edge_attr.size(1)):
            if self.data.edge_attr[:, i:].sum() == self.data.edge_attr.size(0):
                return self.data.edge_attr.size(1) - i
        return 0

    @property
    def num_edge_attributes(self) -> int:
        if self.data.edge_attr is None:
            return 0
        return self.data.edge_attr.size(1) - self.num_edge_labels

    @property
    def raw_file_names(self) -> List[str]:
        names = ['A', 'graph_indicator']
        return [f'{self.name}_{name}.txt' for name in names]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        url = self.cleaned_url if self.cleaned else self.url
        folder = osp.join(self.root, self.name)
        path = download_url(f'{url}/{self.name}.zip', folder)
        extract_zip(path, folder)
        os.unlink(path)
        shutil.rmtree(self.raw_dir)
        os.rename(osp.join(folder, self.name), self.raw_dir)

    def process(self):
        #self.data, self.slices = read_tu_data(self.raw_dir, self.name)
        self.data, self.slices, self.sizes = read_tu_data(self.raw_dir, self.name)
        if self.name in['Fingerprint',"FRANKENSTEIN",'IMDB-BINARY','REDDIT-BINARY',"COLLAB","TRIANGLES"]:
            max_degree, unique_degrees = get_unique_degrees(self)
            self.num_feature = len(unique_degrees)+1
            transform = MyOneHotDegree(max_degree=len(unique_degrees), unique_set=unique_degrees)        
            node_count = 0
            slices_x = [0] 
            for data in self:
                transform(data)
                node_count += data.num_nodes
                slices_x.append(node_count)
            transform(self.data)
            self.slices['x'] = torch.tensor(slices_x)
        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'


def get_TUDataset(dataset, pre_transform):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'TU')
    dataset = TUDataset(path, name=dataset, pre_transform=pre_transform)

    n_feat, n_class = max(dataset.num_feature, 1), dataset.num_classes


    return dataset, n_feat, n_class


def shuffle(dataset, c_train_num, c_val_num, y):
    classes = torch.unique(y)  
    indices = []

    # 首先收集每个类别的索引
    for i in range(len(classes)):
        index = torch.nonzero(y == classes[i]).view(-1)
        index_list = index.tolist()  
        random.shuffle(index_list)  # 打乱每个类别内部的索引
        indices.append(index_list)  # 存储打乱后的索引

    # 分别为训练集、验证集、测试集创建索引列表
    train_index, val_index, test_index = [], [], []

    # 根据每个类别指定的数量分配索引
    for idx_list, cls in zip(indices, classes):
        train_end = c_train_num[cls.item()]  # 训练集索引结束点
        val_end = train_end + c_val_num[cls.item()]  # 验证集索引结束点

        train_index.extend(idx_list[:train_end])
        val_index.extend(idx_list[train_end:val_end])
        test_index.extend(idx_list[val_end:])  # 剩余的都是测试集索引

    # 将列表索引转换为torch.Tensor
    train_index = torch.tensor(train_index, dtype=torch.long)
    val_index = torch.tensor(val_index, dtype=torch.long)
    test_index = torch.tensor(test_index, dtype=torch.long)

    # 根据索引获取数据子集
    train_dataset = dataset[train_index]
    val_dataset = dataset[val_index]
    test_dataset = dataset[test_index]

    return train_dataset, val_dataset, test_dataset

def get_class_num(imb_ratio, num_train, num_val,data_name,n_class,n_data):
    if data_name in ['REDDIT-MULTI-5K', "COLLAB","TRIANGLES","Fingerprint"]:
        n_data_tensor = torch.tensor(n_data)
        sorted_n_data, indices = torch.sort(n_data_tensor, descending=True)   

        # 确定最大类和最小类的数量
        max_class_num = num_train / (1 + (n_class - 2) + (1 / imb_ratio))
        min_class_num = max_class_num / imb_ratio  # 最小类是最大类的1/imb_ratio

        # 计算中间类别的数量，形成等比数列
        mu = (min_class_num / max_class_num) ** (1 / (n_class - 1))
        class_num_list = [round(max_class_num)] + [round(max_class_num * (mu ** i)) for i in range(1, n_class - 1)] \
        + [round(min_class_num)]

        # 计算当前数量总和与目标数量 num_train 的比率
        current_total = sum(class_num_list)
        scale_factor = num_train / current_total

        # 调整所有类别的数量以确保总和为 num_train
        class_num_list = [round(num * scale_factor) for num in class_num_list]
        class_num_list[0] = round(class_num_list[-1] * imb_ratio)
        # 使用 indices 的逆来恢复原始排序
        inv_indices = torch.argsort(indices)
        original_class_num_list = [class_num_list[inv_indices[i]] for i in range(n_class)]
        c_train_num = original_class_num_list
    elif data_name in['FRANKENSTEIN']:
        c_train_num = [int(imb_ratio * num_train),num_train -int(imb_ratio * num_train)]
    else:
        c_train_num = [num_train -int(imb_ratio * num_train),int(imb_ratio * num_train)]

    c_val_num = [num_val] * n_class
    print(c_train_num)
    print(c_val_num)

    return c_train_num, c_val_num

def shuffle(dataset, c_train_num, c_val_num, y):
    num_samples = len(dataset)  # 数据集总样本数
    classes = torch.unique(y)  
    indices = []

    # 首先收集每个类别的索引
    for i in range(len(classes)):
        index = torch.nonzero(y == classes[i]).view(-1)
        index_list = index.tolist()  
        random.shuffle(index_list)  # 打乱每个类别内部的索引
        indices.append(index_list)  # 存储打乱后的索引

    # 分别为训练集、验证集、测试集创建索引列表
    train_index, val_index, test_index = [], [], []

    # 根据每个类别指定的数量分配索引
    for idx_list, cls in zip(indices, classes):
        train_end = c_train_num[cls.item()]  # 训练集索引结束点
        val_end = train_end + c_val_num[cls.item()]  # 验证集索引结束点

        train_index.extend(idx_list[:train_end])
        val_index.extend(idx_list[train_end:val_end])
        test_index.extend(idx_list[val_end:])  # 剩余的都是测试集索引

    # 创建掩码
    train_mask = torch.zeros(num_samples, dtype=torch.bool)
    val_mask = torch.zeros(num_samples, dtype=torch.bool)
    test_mask = torch.zeros(num_samples, dtype=torch.bool)

    train_mask[train_index] = True
    val_mask[val_index] = True
    test_mask[test_index] = True

    return train_mask, val_mask, test_mask

def ana():
    
    return