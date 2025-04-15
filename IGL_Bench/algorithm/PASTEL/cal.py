import networkx as nx
import numpy as np
import multiprocessing as mp
import math
import torch

def cal_shortest_path_distance(adj, approximate, n_nodes):
    Adj = adj.detach().cpu().numpy()
    G = nx.from_numpy_array(Adj)
    G.edges(data=True)
    dists_array = np.zeros((n_nodes, n_nodes))
    dists_dict = all_pairs_shortest_path_length_parallel(G, cutoff=approximate if approximate > 0 else None)

    cnt_disconnected = 0

    for i, node_i in enumerate(G.nodes()):
        shortest_dist = dists_dict[node_i]
        for j, node_j in enumerate(G.nodes()):
            dist = shortest_dist.get(node_j, -1)
            if dist == -1:
                cnt_disconnected += 1
            if dist != -1:
                dists_array[node_i, node_j] = dist
    return dists_array

def all_pairs_shortest_path_length_parallel(graph, cutoff=None, num_workers=4):
    nodes = list(graph.nodes)
    if len(nodes) < 50:
        num_workers = int(num_workers / 4)
    elif len(nodes) < 400:
        num_workers = int(num_workers / 2)

    pool = mp.Pool(processes=num_workers)
    results = [pool.apply_async(single_source_shortest_path_length_range,
                                args=(graph, nodes[int(len(nodes) / num_workers * i):int(len(nodes) / num_workers * (i + 1))], cutoff)) for i in range(num_workers)]
    output = [p.get() for p in results]
    dists_dict = merge_dicts(output)
    pool.close()
    pool.join()
    return dists_dict

def single_source_shortest_path_length_range(graph, node_range, cutoff):
    dists_dict = {}
    for node in node_range:
        dists_dict[node] = nx.single_source_shortest_path_length(graph, node, cutoff)   # unweighted
    return dists_dict



def merge_dicts(dicts):
    result = {}
    for dictionary in dicts:
        result.update(dictionary)
    return result

def cal_group_pagerank_args(pagerank_before, pagerank_after, num_nodes):
    node_pair_group_pagerank_mat = rank_group_pagerank(pagerank_before, pagerank_after, num_nodes) # rank
    PI = 3.1415926
    for i in range(num_nodes):
        for j in range(num_nodes):
            node_pair_group_pagerank_mat[i][j] = 2 - (math.cos((node_pair_group_pagerank_mat[i][j] / (num_nodes * num_nodes)) * PI) + 1)

    return node_pair_group_pagerank_mat

def rank_group_pagerank(pagerank_before, pagerank_after, num_nodes):
    pagerank_dist = torch.mm(pagerank_before, pagerank_after.transpose(-1, -2)).detach().cpu()
    node_pair_group_pagerank_mat = np.zeros((num_nodes, num_nodes))
    node_pair_group_pagerank_mat_list = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            node_pair_group_pagerank_mat_list.append(pagerank_dist[i, j])
    node_pair_group_pagerank_mat_list = np.array(node_pair_group_pagerank_mat_list)
    index = np.argsort(-node_pair_group_pagerank_mat_list)
    rank = np.argsort(index)
    rank = rank + 1
    iter = 0
    for i in range(num_nodes):
        for j in range(num_nodes):
            node_pair_group_pagerank_mat[i][j] = rank[iter]
            iter = iter + 1

    return node_pair_group_pagerank_mat

