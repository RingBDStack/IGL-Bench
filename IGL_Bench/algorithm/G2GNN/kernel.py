import torch
import os

from grakel.kernels import ShortestPath
from grakel import Graph

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

def get_kernel_knn(dataname, kernel_type, knn_nei_num, dataset):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    kernel_file = os.path.join(current_dir, '../../../G2GNN_kernel', 
                               f'{dataname}_{kernel_type}_{knn_nei_num}.txt')

    if(os.path.exists(kernel_file)):
        kernel_simi = torch.load(kernel_file)
    else:
        #dataset = fetch_dataset(dataname, verbose=False)    
        G = [pyg_to_grakel(graph) for graph in dataset]
        if(dataname in ['IMDB-BINARY', 'REDDIT-BINARY']):
            gk = ShortestPath(normalize=True, with_labels=False)
        else:
            gk = ShortestPath(normalize=True)
        kernel_simi = torch.tensor(gk.fit_transform(G))
        torch.save(kernel_simi, kernel_file)

    kernel_idx = torch.topk(kernel_simi, k=knn_nei_num,
                            dim=1, largest=True)[1][:, 1:]

    knn_edge_index = construct_knn(kernel_idx)

    return kernel_idx, knn_edge_index