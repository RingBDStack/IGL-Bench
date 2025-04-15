import os.path as osp
import torch

from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.io import read_tu_data
from typing import Optional, Callable, List
import numpy as np
import os
import shutil
import torch.nn.functional as F
import pandas as pd
import random

from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree, to_networkx
from torch_geometric.data import Data

from torch_geometric.datasets import OGB_MAG
from torch_geometric.data import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator


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
        self.data, self.slices = torch.load(self.processed_paths[0],weights_only=False)
        if self.data.x is not None and not use_node_attr:
            num_node_attributes = self.num_node_attributes
            self.data.x = self.data.x[:, num_node_attributes:]
        if self.data.edge_attr is not None and not use_edge_attr:
            num_edge_attributes = self.num_edge_attributes
            self.data.edge_attr = self.data.edge_attr[:, num_edge_attributes:]
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
            # self.num_feature = len(unique_degrees)
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


class OGBDataset(InMemoryDataset):
    def __init__(self, name, root = 'dataset', transform=None, pre_transform = None, meta_dict = None):
        '''
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - transform, pre_transform (optional): transform/pre-transform graph objects

            - meta_dict: dictionary that stores all the meta-information about data. Default is None, 
                    but when something is passed, it uses its information. Useful for debugging for external contributers.
        ''' 

        self.name = name ## original name, e.g., ogbg-molhiv
        
        if meta_dict is None:
            self.dir_name = '_'.join(name.split('-')) 
            
            # check if previously-downloaded folder exists.
            # If so, use that one.
            if osp.exists(osp.join(root, self.dir_name + '_pyg')):
                self.dir_name = self.dir_name + '_pyg'

            self.original_root = root
            self.root = osp.join(root, self.dir_name)
            
            master = pd.read_csv(os.path.join(os.path.dirname(__file__), 'master.csv'), index_col=0, keep_default_na=False)
            if not self.name in master:
                error_mssg = 'Invalid dataset name {}.\n'.format(self.name)
                error_mssg += 'Available datasets are as follows:\n'
                error_mssg += '\n'.join(master.keys())
                raise ValueError(error_mssg)
            self.meta_info = master[self.name]
            
        else:
            self.dir_name = meta_dict['dir_path']
            self.original_root = ''
            self.root = meta_dict['dir_path']
            self.meta_info = meta_dict
        
        # check version
        # First check whether the dataset has been already downloaded or not.
        # If so, check whether the dataset version is the newest or not.
        # If the dataset is not the newest version, notify this to the user. 
        if osp.isdir(self.root) and (not osp.exists(osp.join(self.root, 'RELEASE_v' + str(self.meta_info['version']) + '.txt'))):
            print(self.name + ' has been updated.')
            if input('Will you update the dataset now? (y/N)\n').lower() == 'y':
                shutil.rmtree(self.root)

        self.download_name = self.meta_info['download_name'] ## name of downloaded file, e.g., tox21

        self.num_tasks = int(self.meta_info['num tasks'])
        self.eval_metric = self.meta_info['eval metric']
        self.task_type = self.meta_info['task type']
        self.__num_classes__ = int(self.meta_info['num classes'])
        self.binary = self.meta_info['binary'] == 'True'

        super(PygGraphPropPredDataset, self).__init__(self.root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])
        
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

def get_class_num(imb_ratio, num_train, num_val, data_name, n_class, n_data):
    if data_name in ['REDDIT-MULTI-5K', "COLLAB","TRIANGLES","Fingerprint"]:
        if imb_ratio == 0.9:
            imb_ratio = 100
        elif imb_ratio == 0.7:
            imb_ratio = 20
        elif imb_ratio == 0.5:
            imb_ratio = 1
        
        n_data_tensor = torch.tensor(n_data)
        sorted_n_data, indices = torch.sort(n_data_tensor, descending=True)   

        max_class_num = num_train / (1 + (n_class - 2) + (1 / imb_ratio))
        min_class_num = max_class_num / imb_ratio 

        mu = (min_class_num / max_class_num) ** (1 / (n_class - 1))
        class_num_list = [round(max_class_num)] + [round(max_class_num * (mu ** i)) for i in range(1, n_class - 1)] \
        + [round(min_class_num)]
        
        current_total = sum(class_num_list)
        scale_factor = num_train / current_total

        class_num_list = [round(num * scale_factor) for num in class_num_list]
        class_num_list[0] = round(class_num_list[-1] * imb_ratio)
        
        inv_indices = torch.argsort(indices)
        original_class_num_list = [class_num_list[inv_indices[i]] for i in range(n_class)]
        c_train_num = original_class_num_list
    elif data_name in['FRANKENSTEIN']:
        c_train_num = [int(imb_ratio * num_train),num_train -int(imb_ratio * num_train)]
    elif data_name in ['ogbg-molhiv']:
        c_val_num = [400] * n_class
        c_test_num = [400] * n_class
        c_train_num = [n - v - t for n, v, t in zip(n_data, c_val_num, c_test_num)]
        return c_train_num, c_val_num
    else:
        c_train_num = [num_train -int(imb_ratio * num_train),int(imb_ratio * num_train)]

    c_val_num = [num_val] * n_class
    # print(c_train_num)
    # print(c_val_num)

    return c_train_num, c_val_num

def shuffle(dataset, c_train_num, c_val_num, y):
    classes = torch.unique(y)  
    indices = []

    for i in range(len(classes)):
        index = torch.nonzero(y == classes[i]).view(-1)
        index_list = index.tolist()  
        random.shuffle(index_list) 
        indices.append(index_list) 

    train_index, val_index, test_index = [], [], []


    for idx_list, cls in zip(indices, classes):
        train_end = c_train_num[cls.item()] 
        val_end = train_end + c_val_num[cls.item()]  

        train_index.extend(idx_list[:train_end])
        val_index.extend(idx_list[train_end:val_end])
        test_index.extend(idx_list[val_end:])  

    train_index = torch.tensor(train_index, dtype=torch.long)
    val_index = torch.tensor(val_index, dtype=torch.long)
    test_index = torch.tensor(test_index, dtype=torch.long)

    return train_index.numpy(), val_index.numpy(), test_index.numpy()
