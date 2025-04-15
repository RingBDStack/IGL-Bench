import numpy as np
from torch_geometric.utils.dropout import dropout_adj
import torch
import random

def remove_edge(edge_index, drop_ratio):
    edge_index, _ = dropout_adj(edge_index, p = drop_ratio)

    return edge_index


def drop_node(x, drop_ratio):
    node_num, _ = x.size()
    drop_num = int(node_num * drop_ratio)

    idx_mask = np.random.choice(node_num, drop_num, replace = False).tolist()

    x[idx_mask] = 0

    return x

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

