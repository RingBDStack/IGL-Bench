import numpy as np
import scipy.sparse as sp
import torch


def encode_onehot(labels):
    """Encode label to a one-hot vector."""
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def row_normalize(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv @ mx
    return mx


def symmetric_normalize(mat):
    """Symmetric-normalize sparse matrix."""
    D = np.asarray(mat.sum(axis=0).flatten())
    D = np.divide(1, D, out=np.zeros_like(D), where=D != 0)
    D = sp.diags(np.asarray(D)[0, :])
    D.data = np.sqrt(D.data)
    return D @ mat @ D


def accuracy(output, labels):
    """Calculate accuracy."""
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def matrix2tensor(mat):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    mat = mat.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((mat.row, mat.col)).astype(np.int64))
    values = torch.from_numpy(mat.data)
    shape = torch.Size(mat.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def tensor2matrix(t):
    """Convert a torch sparse tensor to a scipy sparse matrix."""
    indices = t.indices()
    row, col = indices[0, :].cpu().numpy(), indices[1, :].cpu().numpy()
    values = t.values().cpu().numpy()
    mat = sp.coo_matrix((values, (row, col)), shape=(t.shape[0], t.shape[1]))
    return mat


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)
    return sparse_mx


def random_split(dataset):
    # initialization
    mask = torch.empty(dataset.num_nodes, dtype=torch.bool).fill_(False)
    if dataset.is_ratio:
        num_train = int(dataset.ratio_train * dataset.num_nodes)
        num_val = int(dataset.ratio_val * dataset.num_nodes)
        num_test = dataset.num_nodes - num_train - num_val
    else:
        num_train = dataset.num_train
        num_val = dataset.num_val
        num_test = dataset.num_test

    # get indices for training
    if not dataset.is_ratio and dataset.split_by_class:
        train_idx = dataset.get_split_by_class(num_train_per_class=num_train)
    else:
        train_idx = torch.randperm(dataset.num_nodes)[:num_train]

    # get remaining indices
    mask[train_idx] = True
    remaining = (~mask).nonzero(as_tuple=False).view(-1)
    remaining = remaining[torch.randperm(remaining.size(0))]

    # get indices for validation and test
    val_idx = remaining[:num_val]
    test_idx = remaining[num_val:num_val + num_test]

    return {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }
