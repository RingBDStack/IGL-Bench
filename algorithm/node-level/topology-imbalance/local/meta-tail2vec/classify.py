import argparse

import dgl
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, balanced_accuracy_score
from torch_geometric.graphgym import optim

from model import TwoLayerNN


def train(criterion, optimizer, m, data, label, targets):
    outputs = m(data)
    loss = criterion(outputs, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    indices = np.argmax(outputs.detach().cpu().numpy(), axis=1)

    # print(f"acc:{accuracy};bacc:{bacc};f1:{f1};roc_auc:{roc_auc}")

    print(f"Training Loss: {loss}")


def evaluate(model, inputs, label, onehot, best, baccs, f1s, rocaucs):
    model.eval()

    with torch.no_grad():
        outputs = model(inputs)

    indices = np.argmax(outputs.cpu().numpy(), axis=1)
    label = label.cpu()
    accuracy = accuracy_score(label, indices)
    f1 = f1_score(label, indices, average='macro')
    bacc = balanced_accuracy_score(label, indices)
    roc_auc = roc_auc_score(onehot.cpu(), outputs.cpu().numpy(), multi_class="ovo")

    if accuracy > best:
        best = accuracy
        baccs = bacc
        rocaucs = roc_auc
        f1s = f1


    print(f"acc:{accuracy};bacc:{bacc};f1:{f1};roc_auc:{roc_auc}")

    return best, baccs, f1s, rocaucs


def encode_onehot(labels):
    l_c = np.bincount(label_raw)
    num_classes = len(l_c)
    labels_onehot = np.eye(num_classes)[labels]
    return labels_onehot


if __name__ == '__main__':
    # TODO
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default='mid', help='type:lower higher mid')
    parser.add_argument("--dataset_name", type=str, default='photo', help='dataset name')
    parser.add_argument("--seed", type=int, default=1, help='dataset name')
    args = parser.parse_args()
    dataset_name = args.dataset_name
    t = args.type
    data_range = 0
    print(args)

    masks = torch.load(f'../data_mask/{dataset_name}_{t}_.pth')

    if dataset_name == 'cora':
        dataset = dgl.data.CoraGraphDataset()
        g = dataset[0]
        data_range = 2
    elif dataset_name == 'citeseer':
        dataset = dgl.data.CiteseerGraphDataset()
        g = dataset[0]
        data_range = 1
    elif dataset_name == 'pubmed':
        dataset = dgl.data.PubmedGraphDataset()
        g = dataset[0]
        data_range = 1
    elif dataset_name == 'squirrel' or dataset_name == 'chameleon':
        with open(f'./data/{dataset_name}/out1_node_feature_label.txt', 'r') as file:
            lines = file.readlines()
        feature = []
        labels = []

        for line in lines[1:]:
            parts = line.strip().split('\t')
            node_id = parts[0]
            feature_str = parts[1]
            label = int(parts[2])
            feature_list = list(map(int, feature_str.split(',')))
            feature.append(feature_list)
            labels.append(label)

        features = np.array(feature)
        label_raw = np.array(labels)

        nodes = np.arange(0, len(features))

        with open(f'./data/{dataset_name}/out1_graph_edges.txt', 'r') as file:
            l = file.readlines()
        src, dst = [], []
        for line in l[1:]:
            parts = line.strip().split('\t')
            src.append(int(parts[0]))
            dst.append(int(parts[1]))
        data_range = 3

    elif dataset_name == 'actor':
        with open(f'./data/{dataset_name}/out1_graph_edges.txt', 'r') as file:
            l = file.readlines()
        src, dst = [], []
        for line in l[1:]:
            parts = line.strip().split('\t')
            src.append(int(parts[0]))
            dst.append(int(parts[1]))

        with open(f'./data/{dataset_name}/out1_node_feature_label.txt', 'r') as file:
            lines = file.readlines()
        feature = []
        labels = []

        for line in lines[1:]:
            parts = line.strip().split('\t')
            node_id = parts[0]
            feature_str = parts[1]
            label = int(parts[2])
            feature_list = np.zeros(931)
            indexs = feature_str.split(',')
            for index in indexs:
                feature_list[int(index) - 1] = 1
            feature.append(feature_list)
            labels.append(label)

        features = np.array(feature)
        label_raw = np.array(labels)

        nodes = np.arange(0, len(features))
        data_range = 4

    elif dataset_name == 'computer':
        dataset = dgl.data.AmazonCoBuyComputerDataset()
        g = dataset[0]
        data_range = 16
    elif dataset_name == 'photo':
        dataset = dgl.data.AmazonCoBuyPhotoDataset()
        g = dataset[0]
        data_range = 2
    elif dataset_name == 'arxiv':
        from ogb.nodeproppred import DglNodePropPredDataset
        dataset = DglNodePropPredDataset(name='ogbn-arxiv')
        g = dataset[0][0]
        g.ndata['label'] = dataset[0][1].squeeze()
        from sklearn.preprocessing import StandardScaler
        norm = StandardScaler()
        norm.fit(g.ndata['feat'])
        g.ndata['feat'] = torch.tensor(norm.transform(g.ndata['feat'])).float()

        if t == 'mid':
            data_range = 3
        elif t == 'lower':
            data_range = 0
        elif t == 'higher':
            data_range = 1

    if dataset_name == 'cora' or dataset_name == 'citeseer' or dataset_name == 'pubmed' or dataset_name == 'computer' or dataset_name == 'photo' or dataset_name == 'arxiv':
        g = g.remove_self_loop()
        features = g.ndata["feat"].numpy()
        label_raw = g.ndata["label"]
        src, dst = g.edges()
        src = src.numpy()
        dst = dst.numpy()
        nodes = g.nodes().numpy()

    # features = torch.load('./data/' + dataset_name + '/embs.pth')

    train_mask = masks['train_mask']
    val_mask = masks['val_mask']
    test_mask = masks['test_mask']

    one_hot = encode_onehot(label_raw)
    train_data = features[train_mask]
    train_labels = label_raw[train_mask]
    train_onehot = encode_onehot(train_labels)
    if dataset_name == 'squirrel' or dataset_name == 'chameleon' or dataset_name == 'actor':
        train_labels_cuda = torch.from_numpy(train_labels.astype(np.float32)).cuda()
    else:
        train_labels_cuda = train_labels.cuda()

    test_node_labels = label_raw[test_mask]
    print(len(test_node_labels))
    test_node_onehot = encode_onehot(test_node_labels)
    test_node_onehot = torch.from_numpy(test_node_onehot).cuda()

    if dataset_name == 'squirrel' or dataset_name == 'chameleon' or dataset_name == 'actor':
        test_labels_cuda = torch.from_numpy(test_node_labels.astype(np.float32)).cuda()
    else:
        test_labels_cuda = test_node_labels.cuda()
    

    train_data = torch.from_numpy(train_data.astype(np.float32)).cuda()
    train_onehot = torch.from_numpy(train_onehot).cuda()

    l_c = np.bincount(label_raw)
    num_classes = len(l_c)

    accscores, baccscores, f1scores, rocaucscores = [], [], [], []

    iter_num = args.seed
    
    all_embeddings = []
    node_ids = []
    with open(f'./data/{dataset_name}/{t}_result_{iter_num}.csv', 'r', encoding='utf-8') as file:
        for line in file:
            temp = list(line.strip('\n').split(' '))
            node = int(temp[0])
            emb = temp[1:]
            embeddings = [float(item) for item in emb]
            all_embeddings.append(embeddings)
            node_ids.append(int(temp[0]))

    if data_range == 0:
        embeddings = all_embeddings
    else:
        embeddings = all_embeddings[:-data_range]

    model = TwoLayerNN(input_dim=features.shape[1], output_dim=num_classes).cuda()
    print(model)

    combined = list(zip(node_ids, embeddings))
    sorted_combined = sorted(combined, key=lambda x: x[0])
    sorted_node_ids, sorted_embeddings = zip(*sorted_combined)
    test_data = np.array(list(sorted_embeddings))

    test_data = torch.from_numpy(test_data.astype(np.float32)).cuda()
    # test_data = torch.from_numpy(features[test_mask])

    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    best_acc = 0.0
    bbacc, bf1, broc = 0.0, 0.0, 0.0

    for i in range(500):
        train(criterion, optimizer, model, train_data, train_labels_cuda, train_onehot)
        best_acc, bbacc, bf1, broc = evaluate(model, test_data, test_labels_cuda, test_node_onehot, best_acc, bbacc,
                                              bf1, broc)

    print(f"BEST RESULT;acc:{best_acc};bacc:{bbacc};f1:{bf1};roc_auc:{broc}")

