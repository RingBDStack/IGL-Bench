import numpy as np
import random
import torch
import scipy.sparse as sp
from torch import nn
from torch_geometric.utils import add_remaining_self_loops, degree
from torch_scatter import scatter_max, scatter_add, scatter
import os

from grakel.datasets import fetch_dataset
from grakel.kernels import ShortestPath
from grakel import Graph
import sys


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def construct_knn(kernel_idx):
    edge_index = [[], []]

    for i in range(len(kernel_idx)):
        for j in range(len(kernel_idx[i])):
            edge_index[0].append(kernel_idx[i, j].item())
            edge_index[1].append(i)

            edge_index[1].append(kernel_idx[i, j].item())
            edge_index[0].append(i)

    return torch.tensor(edge_index, dtype=torch.long)

def pyg_to_grakel(pyg_graph):
    edge_index = pyg_graph.edge_index.numpy()
    edges = list(zip(edge_index[0], edge_index[1]))
    node_labels = {i: str(label) for i, label in enumerate(pyg_graph.x.numpy())}
    return Graph(edges, node_labels=node_labels)

def get_kernel_knn(dataname, kernel_type, knn_nei_num,dataset):
    kernel_file = './kernel/' + dataname + '_' + \
        kernel_type + '_' + str(knn_nei_num) + '.txt'

    if(os.path.exists(kernel_file)):
        kernel_simi = torch.load(kernel_file)
    else:
        #dataset = fetch_dataset(dataname, verbose=False)    
        G = [pyg_to_grakel(graph) for graph in dataset]
        if(dataset.name in ['IMDB-BINARY', 'REDDIT-BINARY']):
            gk = ShortestPath(normalize=True, with_labels=False)
        else:
            gk = ShortestPath(normalize=True)
        kernel_simi = torch.tensor(gk.fit_transform(G))
        torch.save(kernel_simi, kernel_file)
        sys.exit()

    kernel_idx = torch.topk(kernel_simi, k=knn_nei_num,
                            dim=1, largest=True)[1][:, 1:]

    knn_edge_index = construct_knn(kernel_idx)

    return kernel_idx, knn_edge_index


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


def upsample(dataset):
    y = torch.tensor([dataset[i].y for i in range(len(dataset))])
    classes = torch.unique(y)

    num_class_graph = [(y == i.item()).sum() for i in classes]

    max_num_class_graph = max(num_class_graph)

    chosen = []
    for i in range(len(classes)):
        train_idx = torch.where((y == classes[i]) == True)[0].tolist()

        up_sample_ratio = max_num_class_graph / num_class_graph[i]
        up_sample_num = int(
            num_class_graph[i] * up_sample_ratio - num_class_graph[i])

        if(up_sample_num <= len(train_idx)):
            up_sample = random.sample(train_idx, up_sample_num)
        else:
            tmp = int(up_sample_num / len(train_idx))
            up_sample = train_idx * tmp
            tmp = up_sample_num - len(train_idx) * tmp

            up_sample.extend(random.sample(train_idx, tmp))

        chosen.extend(up_sample)
    
    if not chosen:
        return list(dataset)
    
    chosen = torch.tensor(chosen)
    extend_data = dataset[chosen]

    data = list(dataset) + list(extend_data)

    return data


def find_knn_id(batch_id, kernel_idx):
    knn_id = set(kernel_idx[batch_id].view(-1).tolist())
    pad_knn_id = knn_id.difference(set(batch_id.tolist()))

    return list(pad_knn_id)


def batch_to_gpu(batch, device):
    for key in batch:
        if isinstance(batch[key], list):
            for i in range(len(batch[key])):
                batch[key][i] = batch[key][i].to(device)
        else:
            batch[key] = batch[key].to(device)

    return batch


def consis_loss(logps, temp=0.5):
    ps = [torch.exp(p) for p in logps]
    sum_p = 0.
    for p in ps:
        sum_p = sum_p + p
    avg_p = sum_p / len(ps)

    sharp_p = (torch.pow(avg_p, 1. / temp) /
               torch.sum(torch.pow(avg_p, 1. / temp), dim=1, keepdim=True)).detach()
    loss = 0.
    for p in ps:
        loss += torch.mean((p - sharp_p).pow(2).sum(1))
    loss = loss / len(ps)
    return 1 * loss



def embed_smote(embed, num_training_graph, y, k):
    max_num_training_graph = max(num_training_graph)
    classes = torch.unique(y)

    embed_aug = []
    y_aug = []

    for i in range(len(classes)):
        train_idx = torch.where((y == classes[i]) == True)[0].tolist()

        c_embed = embed[train_idx]
        c_dist = torch.cdist(c_embed, c_embed, p=2)

        # different from original smote, we also include itself in case of no other nodes to use
        c_min_idx = torch.topk(c_dist, min(k, c_dist.size(0)), largest=False)[
            1][:, :].tolist()

        up_sample_ratio = max_num_training_graph / \
            num_training_graph[i]
        up_sample_num = int(
            num_training_graph[i] * up_sample_ratio - num_training_graph[i])

        tmp = 1
        head_list = list(np.arange(0, len(train_idx)))

        while(tmp <= up_sample_num):
            head_id = random.choice(head_list)
            tail_id = random.choice(c_min_idx[head_id])

            delta = torch.rand(1).to(c_embed.device)
            new_embed = torch.lerp(
                c_embed[head_id], c_embed[tail_id], delta)
            embed_aug.append(new_embed)
            y_aug.append(classes[i])

            tmp += 1

    if(embed_aug == []):
        return embed, y

    return torch.stack(embed_aug), torch.stack(y_aug).to(embed.device)


def homophily(edge_index, y):
    degree_cal = degree(edge_index[1], num_nodes=y.size(0))

    edge_homo = (y[edge_index[0]] == y[edge_index[1]]
                 ).sum().item() / edge_index.size(1)

    tmp = y[edge_index[0]] == y[edge_index[1]]
    node_homo = scatter(tmp, edge_index[1], dim=0, dim_size=y.size(
        0), reduce='add') / degree_cal

    return edge_homo, node_homo.mean()
