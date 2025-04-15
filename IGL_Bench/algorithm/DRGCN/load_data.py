import numpy as np
import random
import scipy.sparse as sp
import torch

def _torch_sparse_to_scipy(tsp, shape=None):
    tsp = tsp.coalesce()
    idx = tsp.indices().cpu().numpy()
    val = tsp.values().cpu().numpy()
    if shape is None:
        shape = tsp.shape
    return sp.coo_matrix((val, (idx[0], idx[1])), shape=shape).tocsr()

def _dense_to_scipy(mat_like, shape=None):
    """torch / numpy 稠密矩阵 → CSR"""
    if isinstance(mat_like, torch.Tensor):
        mat_like = mat_like.cpu().numpy()
    if shape is None:
        shape = mat_like.shape
    return sp.coo_matrix(mat_like.reshape(shape)).tocsr()

def _edge_index_to_scipy(edge_index, num_nodes, edge_weight=None):
    row, col = edge_index.cpu().numpy()
    if edge_weight is None:
        edge_weight = np.ones(row.shape[0], dtype=np.float32)
    else:
        edge_weight = edge_weight.cpu().numpy()
    return sp.coo_matrix((edge_weight, (row, col)),
                         shape=(num_nodes, num_nodes)).tocsr()

def _any_adj_to_scipy(adj_like, num_nodes):
    """
    将任意 PyG 里可能出现的邻接存储格式统一转 CSR
    """
    # 1) 直接是 scipy
    if isinstance(adj_like, sp.spmatrix):
        return adj_like.tocsr()

    # 2) torch 稀疏张量
    if isinstance(adj_like, torch.Tensor):
        if adj_like.is_sparse:
            return _torch_sparse_to_scipy(adj_like, (num_nodes, num_nodes))
        else:                       # 稠密 torch.Tensor
            return _dense_to_scipy(adj_like, (num_nodes, num_nodes))

    # 3) torch_sparse.SparseTensor
    try:
        from torch_sparse import SparseTensor
        if isinstance(adj_like, SparseTensor):
            row, col, val = adj_like.coo()
            return sp.coo_matrix(
                (val.cpu().numpy(),
                 (row.cpu().numpy(), col.cpu().numpy())),
                shape=(num_nodes, num_nodes)).tocsr()
    except ImportError:
        pass

    # 4) numpy ndarray / list 等稠密
    if isinstance(adj_like, (np.ndarray, list)):
        return _dense_to_scipy(np.asarray(adj_like, dtype=np.float32),
                               (num_nodes, num_nodes))

    raise TypeError(f"Unsupported adjacency type: {type(adj_like)}")

# ------------------------------------------------------------------ #
def data_process(dataset):

    # ---------- 基础数据 ----------
    x = dataset.x.cpu().numpy().astype(np.float32)
    label = dataset.y.squeeze().cpu().numpy().astype(np.int64)
    num_nodes = x.shape[0]

    # ---------- 邻接矩阵 ----------
    if hasattr(dataset, 'adj') and dataset.adj is not None:
        adj = _any_adj_to_scipy(dataset.adj, num_nodes)
    elif hasattr(dataset, 'edge_index'):
        edge_weight = getattr(dataset, 'edge_weight', None)
        adj = _edge_index_to_scipy(dataset.edge_index, num_nodes, edge_weight)
    else:
        raise ValueError("Dataset 必须包含 adj / edge_index")

    # ---------- 归一化邻接 ----------
    if hasattr(dataset, 'adj_norm') and dataset.adj_norm is not None:
        adj_norm = _any_adj_to_scipy(dataset.adj_norm, num_nodes)
    else:
        deg = np.array(adj.sum(1)).flatten()
        deg_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
        D_inv_sqrt = sp.diags(deg_inv_sqrt)
        adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt

    # ---------- 划分索引 ----------
    train_indexes = np.asarray(dataset.train_index, dtype=np.int64)
    validation_indexes = np.asarray(dataset.val_index, dtype=np.int64)
    test_indexes = np.asarray(dataset.test_index, dtype=np.int64)

    # ---------- GAN 采样 ----------
    label_counts = {}
    for idx in train_indexes:
        lab = int(label[idx])
        label_counts.setdefault(lab, []).append(idx)
    balance_num = max(len(v) for v in label_counts.values())

    real_gan_nodes, generated_gan_nodes, real_node_sequence = [], [], []
    for lab, nodes in label_counts.items():
        for n in nodes:                       # 全量真实
            real_gan_nodes.append([n, lab])
            real_node_sequence.append(n)
        for _ in range(balance_num - len(nodes)):  # 随机补足
            s = random.choice(nodes)
            real_gan_nodes.append([s, lab])
            real_node_sequence.append(s)
            generated_gan_nodes.append([s, lab])

    perm = np.random.permutation(len(real_gan_nodes))
    real_gan_nodes = [real_gan_nodes[i] for i in perm]
    real_node_sequence = [real_node_sequence[i] for i in perm]

    # ---------- 二部图 ----------
    adj_coo = adj.tocoo()
    neighbor_dict = {}
    for r, c in zip(adj_coo.row, adj_coo.col):
        neighbor_dict.setdefault(r, []).append(c)

    all_neighbor_nodes = sorted(
        {nbr for v in real_node_sequence for nbr in neighbor_dict.get(v, [])})
    real_num = len(real_node_sequence)
    neigh_num = len(all_neighbor_nodes)

    adj_neighbor = np.zeros((real_num, neigh_num), dtype=np.float32)
    col_map = {n: j for j, n in enumerate(all_neighbor_nodes)}
    for i, v in enumerate(real_node_sequence):
        for nbr in neighbor_dict.get(v, []):
            j = col_map.get(nbr)
            if j is not None:
                adj_neighbor[i, j] = 1.0

    # ---------- 返回 ----------
    return (x, adj, adj_norm, label,
            train_indexes, test_indexes, validation_indexes,
            real_gan_nodes, generated_gan_nodes,
            adj_neighbor, all_neighbor_nodes)
