import dgl
import scipy.sparse as sp
import numpy as np
import torch
import utils
from torch_geometric.data import Data, InMemoryDataset
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
import os.path as osp
from torch_geometric.utils.loop import add_self_loops, remove_self_loops
from torch_geometric.datasets import Planetoid, Coauthor, Amazon, WikiCS, CitationFull, Actor

def decide_config(root, dataset):
    """
    Create a configuration to download datasets
    :param root: A path to a root directory where data will be stored
    :param dataset: The name of the dataset to be downloaded
    :return: A modified root dir, the name of the dataset class, and parameters associated to the class
    """
    dataset = dataset.lower()
    if dataset == 'cora' or dataset == 'citeseer' or dataset == "pubmed":
        root = osp.join(root, "pyg", "planetoid")
        params = {"kwargs": {"root": root, "name": dataset, "split": 'full'},
                  "name": dataset, "class": Planetoid, "src": "pyg"}
    elif dataset == 'cora_full':
        dataset = "cora"
        root = osp.join(root, "pyg")
        params = {"kwargs": {"root": root, "name": dataset},
                  "name": dataset, "class": CitationFull, "src": "pyg"}
    elif dataset == 'actor':
        dataset = "Actor"
        root = osp.join(root, "pyg")
        params = {"kwargs": {"root": root},
                  "name": dataset, "class": Actor, "src": "pyg"}
    elif dataset == 'dblp':
        dataset = "DBLP"
        root = osp.join(root, "pyg")
        params = {"kwargs": {"root": root, "name": dataset},
                  "name": dataset, "class": CitationFull, "src": "pyg"}
    elif dataset == "computers":
        dataset = "Computers"
        root = osp.join(root, "pyg")
        params = {"kwargs": {"root": root, "name": dataset},
                  "name": dataset, "class": Amazon, "src": "pyg"}
    elif dataset == "photo":
        dataset = "Photo"
        root = osp.join(root, "pyg")
        params = {"kwargs": {"root": root, "name": dataset},
                  "name": dataset, "class": Amazon, "src": "pyg"}
    elif dataset == "cs" :
        dataset = "CS"
        root = osp.join(root, "pyg")
        params = {"kwargs": {"root": root, "name": dataset},
                  "name": dataset, "class": Coauthor, "src": "pyg"}
    elif dataset == "physics":
        dataset = "Physics"
        root = osp.join(root, "pyg")
        params = {"kwargs": {"root": root, "name": dataset},
                  "name": dataset, "class": Coauthor, "src": "pyg"}
    elif dataset == "wikics":
        dataset = "WikiCS"
        root = osp.join(root, "pyg", 'WikiCS')
        params = {"kwargs": {"root": root},
                  "name": dataset, "class": WikiCS, "src": "pyg"}
    else:
        raise Exception(
            f"Unknown dataset name {dataset}, name has to be one of the following 'cora', 'citeseer', 'pubmed', 'photo', 'computers', 'cs', 'physics', 'actor'")
    return params


def download_pyg_data(config):
    """
    Downloads a dataset from the PyTorch Geometric library

    :param config: A dict containing info on the dataset to be downloaded
    :return: A tuple containing (root directory, dataset name, data directory)
    """
    leaf_dir = config["kwargs"]["root"].split("/")[-1].strip()
    data_dir = osp.join(config["kwargs"]["root"], "" if config["name"] == leaf_dir else config["name"])
    dst_path = osp.join(data_dir, "raw", "data.pt")
    if not osp.exists(dst_path):
        DatasetClass = config["class"]
        dataset = DatasetClass(**config["kwargs"])
        torch.save((dataset.data, dataset.slices), dst_path)
    return config["kwargs"]["root"], config["name"], data_dir

def download_data(root, dataset):
    """
    Download data from different repositories. Currently only PyTorch Geometric is supported

    :param root: The root directory of the dataset
    :param name: The name of the dataset
    :return:
    """
    config = decide_config(root=root, dataset=dataset)
    if config["src"] == "pyg":
        return download_pyg_data(config)

class Dataset(InMemoryDataset):
    """
    A PyTorch InMemoryDataset to build multi-view dataset through graph data augmentation
    """

    def __init__(self, type="mid", root="data", dataset='cora', transform=None, pre_transform=None, is_normalize=True, add_self_loop=False):
        dataset_name = dataset
        masks = torch.load(f'../data_mask/{dataset_name}_{type}_.pth')
        if dataset_name == 'cora':
            dataset = dgl.data.CoraGraphDataset()
            g = dataset[0]
            non_directed = True
        elif dataset_name == 'citeseer':
            dataset = dgl.data.CiteseerGraphDataset()
            g = dataset[0]
            non_directed = True
        elif dataset_name == 'pubmed':
            dataset = dgl.data.PubmedGraphDataset()
            g = dataset[0]
            non_directed = True
        elif dataset_name == 'squirrel':
            dataset = dgl.data.SquirrelDataset()
            g = dataset[0]
            non_directed = False
        elif dataset_name == 'chameleon':
            dataset = dgl.data.ChameleonDataset()
            g = dataset[0]
            non_directed = False
        elif dataset_name == 'actor':
            dataset = dgl.data.ActorDataset()
            g = dataset[0]
            non_directed = False
        elif dataset_name == 'computer':
            dataset = dgl.data.AmazonCoBuyComputerDataset()
            g = dataset[0]
            non_directed = True
        elif dataset_name == 'photo':
            dataset = dgl.data.AmazonCoBuyPhotoDataset()
            g = dataset[0]
            non_directed = True
        elif dataset == 'arxiv':
            from ogb.nodeproppred import DglNodePropPredDataset
            dataset = DglNodePropPredDataset(name='ogbn-arxiv')
            g = dataset[0][0]
            g.ndata['label'] = dataset[0][1].squeeze()
            from sklearn.preprocessing import StandardScaler
            norm = StandardScaler()
            norm.fit(g.ndata['feat'])
            g.ndata['feat'] = torch.tensor(norm.transform(g.ndata['feat'])).float()
            non_directed = True
        else:
            raise ValueError('Dataset not available')

        g = g.remove_self_loop().add_self_loop()
        self.features = g.ndata["feat"]
        self.labels = g.ndata["label"]
        g = dgl.to_bidirected(g, copy_ndata=True)
        self.edge_index = torch.stack(g.edges())
        self.nodes_num = g.num_nodes()

        adj = sp.coo_matrix(
            (np.ones(self.edge_index.shape[1]), (self.edge_index[0, :], self.edge_index[1, :])),
            shape=(self.nodes_num, self.nodes_num), dtype=np.float32)
        self.adj = utils.sparse_mx_to_torch_sparse_tensor(adj)

        self.train_mask = masks['train_mask']
        self.val_mask = masks['val_mask']
        self.test_mask = masks['test_mask']

        if dataset_name == 'computer':
            self.test_mask[261] = True
            self.test_mask[61] = True
            self.test_mask[555] = True

    def process_full_batch_data(self, data):
        """
        Augmented view data generation using the full-batch data.

        :param view1data:
        :return:
        """
        print("Processing full batch data")
        if 'planetoid' in self.root: # for LT dataset
            data = Data(edge_index=data.edge_index, edge_attr=data.edge_attr,
                    x=data.x, y=data.y, num_nodes=data.num_nodes, train_mask=data.train_mask, val_mask=data.val_mask, test_mask=data.test_mask)
        else:
            data = Data(edge_index=data.edge_index, edge_attr=data.edge_attr, x=data.x, y=data.y, num_nodes=data.num_nodes)
        return [data]

    def process(self):
        """
        Process a full batch data.
        :return:
        """
        processed_path = osp.join(self.processed_dir, self.processed_file_names[0])
        if not osp.exists(processed_path):
            path = osp.join(self.raw_dir, self.raw_file_names[0])
            data, _ = torch.load(path)
            edge_attr = data.edge_attr
            edge_attr = torch.ones(data.edge_index.shape[1]) if edge_attr is None else edge_attr
            data.edge_attr = edge_attr
            data_list = self.process_full_batch_data(data)
            data, slices = self.collate(data_list)
            torch.save((data, slices), processed_path)

    @property
    def raw_file_names(self):
        return ["data.pt"]

    @property
    def processed_file_names(self):
        return [f'byg.data.pt']

    @property
    def raw_dir(self):
        return osp.join(self.data_dir, "raw")

    @property
    def processed_dir(self):
        return osp.join(self.data_dir, "processed")

    @property
    def model_dir(self):
        return osp.join(self.data_dir, "model")

    @property
    def result_dir(self):
        return osp.join(self.data_dir, "result")

    @property
    def dirs(self):
        return [self.raw_dir, self.processed_dir, self.model_dir, self.result_dir]

    def download(self):
        pass


