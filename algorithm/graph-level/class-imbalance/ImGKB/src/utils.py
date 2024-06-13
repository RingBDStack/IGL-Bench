import numpy as np
import torch
from math import ceil
from scipy.sparse import csr_matrix,lil_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix,accuracy_score, balanced_accuracy_score, roc_curve
from sklearn.preprocessing import label_binarize

def load_data(dataset):
    graph_indicator = np.loadtxt("datasets/%s/%s_graph_indicator.txt"%(dataset, dataset), dtype=np.int64)
    # the value in the i-th line is the graph_id of the node with node_id i
    edges = np.loadtxt("datasets/%s/%s_A.txt"%(dataset, dataset), dtype=np.int64, delimiter="," )
    # each line correspond to (row, col) resp. (node_id, node_id)
    edges -= 1
    _, graph_size = np.unique(graph_indicator, return_counts=True)
    # _:graph_idx, graph_size:the number of node of a graph
    A = csr_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(graph_indicator.size, graph_indicator.size))#.todense()
    X = np.loadtxt("datasets/%s/%s_node_labels.txt" % (dataset, dataset), dtype=np.int64).reshape(-1, 1)
    # the value in the i-th line corresponds to the node with node_id i
    enc = OneHotEncoder(sparse=False)
    X = enc.fit_transform(X)
    adj = []
    features = []
    start_idx = 0
    for i in range(graph_size.size):
        adj.append(A[start_idx:start_idx + graph_size[i], start_idx:start_idx + graph_size[i]])
        features.append(X[start_idx:start_idx + graph_size[i], :])
        start_idx += graph_size[i]
    class_labels = np.loadtxt("datasets/%s/%s_graph_labels.txt" % (dataset, dataset), dtype=np.int64)
    class_labels = np.where(class_labels==-1, 0, class_labels)
    return adj, features, class_labels

def my_load_data(data_name,imb):
    import os.path as osp
    from data import TUDataset,get_class_num
    import torch_geometric.transforms as T
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..','data', 'TU')
    dataset = TUDataset(path, name=data_name, pre_transform=T.ToSparseTensor(remove_edge_index=False))
    edges = dataset.data.edge_index.numpy()
    graph_indicator = []
    for graph_id, data in enumerate(dataset):
        num_nodes = data.num_nodes
        graph_indicator.extend([graph_id + 1] * num_nodes)
    graph_indicator = np.array(graph_indicator)
    A = csr_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(graph_indicator.size, graph_indicator.size))
    X = dataset.data.x.numpy()
    labels = [data.y.item() for data in dataset]
    _, graph_size = np.unique(graph_indicator, return_counts=True)
    adj = []
    features = []
    start_idx = 0
    for i in range(len(dataset)):
        adj.append(A[start_idx:start_idx + graph_size[i], start_idx:start_idx + graph_size[i]])
        features.append(X[start_idx:start_idx + graph_size[i], :])
        start_idx += graph_size[i]
        
    _, counts = np.unique(labels, return_counts=True)
    labels = np.array(labels)
    num_train = (int)(len(dataset) * 0.1)
    num_val = (int)(len(dataset) * 0.1 / dataset.num_classes)
    train_num,val_num = get_class_num(imb, num_train, num_val,data_name,dataset.num_classes,counts)
    return adj, features, labels ,train_num,val_num
    
    
    

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row,
                                          sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def generate_batches(adj, features, y, batch_size, graph_pooling_type, device,  shuffle=True):
    N = len(y)
    if shuffle:
        index = np.random.permutation(N)
    else:
        index = np.array(range(N), dtype=np.int32)

    n_batches = ceil(N/batch_size)

    adj_lst = list()
    features_lst = list()
    graph_indicator_lst = list()
    y_lst = list()
    graph_pool_lst=list()
    nu=0 ## compute null number
    for i in range(0, N, batch_size):
        n_graphs = min(i+batch_size, N) - i
        n_nodes = sum([adj[index[j]].shape[0] for j in range(i, min(i+batch_size, N))])

        adj_batch = lil_matrix((n_nodes, n_nodes))
        features_batch = np.zeros((n_nodes, features[0].shape[1]))
        graph_indicator_batch = np.zeros(n_nodes)
        y_batch = np.zeros(n_graphs)
        graph_pool_batch = lil_matrix((n_graphs, n_nodes))

        idx = 0
        for j in range(i, min(i+batch_size, N)):
            n = adj[index[j]].shape[0]
            adj_batch[idx:idx+n, idx:idx+n] = adj[index[j]]    
            features_batch[idx:idx+n,:] = features[index[j]]
            graph_indicator_batch[idx:idx+n] = j-i
            y_batch[j-i] = y[index[j]]
            # print(y_batch)
            if graph_pooling_type == "average":
                graph_pool_batch[j-i, idx:idx+n] = 1./n
            else:
                graph_pool_batch[j-i, idx:idx+n] = 1

            idx += n
        if sum(y_batch)==0 or sum(y_batch)==n_graphs:
            nu+=1
            pass
        else:
            adj_lst.append(sparse_mx_to_torch_sparse_tensor(adj_batch).to(device))
            features_lst.append(torch.FloatTensor(features_batch).to(device))
            graph_indicator_lst.append(torch.LongTensor(graph_indicator_batch).to(device))
            y_lst.append(torch.LongTensor(y_batch).to(device))
            graph_pool_lst.append(sparse_mx_to_torch_sparse_tensor(graph_pool_batch).to(device))

    return adj_lst, features_lst, graph_pool_lst, graph_indicator_lst, y_lst, n_batches-nu


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# def compute_metrics(logits, labels):
#     auc_ = roc_auc_score(labels.detach().cpu().numpy(), logits.detach().cpu().numpy()[:, 1])
#     target_names = ['C0', 'C1']
#     DICT = classification_report(labels.detach().cpu().numpy(), logits.detach().cpu().numpy().argmax(axis=1),
#                                  target_names=target_names, output_dict=True)
#     macro_recall = DICT['macro avg']['recall']
#     macro_f1 = DICT['macro avg']['f1-score']
#     return auc_, macro_recall, macro_f1

def compute_metrics(logits, labels):
    logits_np = logits.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    # 计算 AUC
    auc_ = roc_auc_score(labels_np, logits_np[:, 1])

    # 生成分类报告字典
    target_names = ['C0', 'C1']
    DICT = classification_report(labels_np, logits_np.argmax(axis=1),
                                 target_names=target_names, output_dict=True)

    # 从分类报告中提取宏观召回率和 F1 分数
    macro_recall = DICT['macro avg']['recall']
    macro_f1 = DICT['macro avg']['f1-score']

    # 计算精确度和平衡精确度
    acc = accuracy_score(labels_np, logits_np.argmax(axis=-1))
    bacc = balanced_accuracy_score(labels_np, logits_np.argmax(axis=-1))

    return auc_, macro_recall, macro_f1, acc, bacc

def my_metric(logits, labels):
    logits_np = logits.detach().cpu().numpy()
    labels_np = labels
    n_classes = logits_np.shape[1]

    #labels_one_hot = label_binarize(labels_np, classes=range(n_classes))
    labels_onehot = np.eye(n_classes)[labels_np]
    auc_ = roc_auc_score(labels_onehot, logits_np, multi_class='ovr', average='macro')

    target_names = [f'C{i}' for i in range(n_classes)]
    DICT = classification_report(labels_np, logits_np.argmax(axis=1),
                                 target_names=target_names, output_dict=True)

    macro_recall = DICT['macro avg']['recall']
    macro_f1 = DICT['macro avg']['f1-score']

    acc = accuracy_score(labels_np, logits_np.argmax(axis=-1))
    bacc = balanced_accuracy_score(labels_np, logits_np.argmax(axis=-1))

    cm = confusion_matrix(labels_np, logits_np.argmax(axis=-1))
    FPR = {}
    TPR = {}
    for i in range(n_classes):
        TN = cm.sum() - (cm[i,:].sum() + cm[:,i].sum() - cm[i,i])
        FP = cm[:,i].sum() - cm[i,i]
        FN = cm[i,:].sum() - cm[i,i]
        TP = cm[i,i]
        
        FPR[f'C{i}'] = FP / (FP + TN) if (FP + TN) > 0 else 0
        TPR[f'C{i}'] = TP / (TP + FN) if (TP + FN) > 0 else 0
        print(f"Class {i} - FPR: {FPR[f'C{i}']:.2f}, TPR: {TPR[f'C{i}']:.2f}")

    return auc_, macro_recall, macro_f1, acc, bacc  

def compute_metrics_multiclass(logits, labels):
    logits_np = logits.detach().cpu().numpy()
    if not isinstance(labels, np.ndarray):
        labels_np = labels.detach().cpu().numpy()
    else:
        labels_np = labels

    eps = 1e-6  
    logits_np = logits_np + eps * np.random.rand(*logits_np.shape)
    logits_np = logits_np / logits_np.sum(axis=1, keepdims=True)  

    n_classes = logits_np.shape[1]
    labels_binarized = label_binarize(labels_np, classes=range(n_classes))

    # 尝试计算AUC，如果失败则设为0.5
    try:
        auc_ = roc_auc_score(labels_binarized, logits_np, multi_class='ovr', average='macro')
    except ValueError as e:
        auc_ = 0.5  

    actual_classes = np.unique(labels_np)
    target_names = [f'C{i}' for i in range(n_classes)]  

    DICT = classification_report(labels_np, logits_np.argmax(axis=1),
                                 target_names=target_names, labels=range(n_classes), output_dict=True)

    macro_recall = DICT['macro avg']['recall']
    macro_f1 = DICT['macro avg']['f1-score']

    acc = accuracy_score(labels_np, logits_np.argmax(axis=-1))
    bacc = balanced_accuracy_score(labels_np, logits_np.argmax(axis=-1))

    return auc_, macro_recall, macro_f1, acc, bacc

def test_multiclass(logits, labels):
    logits_np = logits.detach().cpu().numpy()
    if not isinstance(labels, np.ndarray):
        labels_np = labels.detach().cpu().numpy()
    else:
        labels_np = labels

    eps = 1e-6  
    logits_np = logits_np + eps * np.random.rand(*logits_np.shape)
    logits_np = logits_np / logits_np.sum(axis=1, keepdims=True) 

    n_classes = logits_np.shape[1]
    labels_binarized = label_binarize(labels_np, classes=range(n_classes))

    try:
        auc_ = roc_auc_score(labels_binarized, logits_np, multi_class='ovr', average='macro')
    except ValueError as e:
        auc_ = 0.5 
    cm = confusion_matrix(labels_np, logits_np.argmax(axis=-1))
    FPR = {}
    TPR = {}

    for i in range(n_classes):
        TN = cm.sum() - (cm[i,:].sum() + cm[:,i].sum() - cm[i,i])
        FP = cm[:,i].sum() - cm[i,i]
        FN = cm[i,:].sum() - cm[i,i]
        TP = cm[i,i]
        
        FPR[f'C{i}'] = FP / (FP + TN) if (FP + TN) > 0 else 0
        TPR[f'C{i}'] = TP / (TP + FN) if (TP + FN) > 0 else 0
        print(f"Class {i} - FPR: {FPR[f'C{i}']:.2f}, TPR: {TPR[f'C{i}']:.2f}")

    actual_classes = np.unique(labels_np)
    target_names = [f'C{i}' for i in range(n_classes)]  

    DICT = classification_report(labels_np, logits_np.argmax(axis=1),
                                 target_names=target_names, labels=range(n_classes), output_dict=True)

    macro_recall = DICT['macro avg']['recall']
    macro_f1 = DICT['macro avg']['f1-score']

    acc = accuracy_score(labels_np, logits_np.argmax(axis=-1))
    bacc = balanced_accuracy_score(labels_np, logits_np.argmax(axis=-1))

    return auc_, macro_recall, macro_f1, acc, bacc
