import dgl
import numpy as np
import torch
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from scipy import sparse
from collections import defaultdict
from sklearn.model_selection import train_test_split

dataset_name = 'cora'
t = 'mid'
nodes_num = 2708
train_dir = './data/' + dataset_name + '/graph.embeddings'
test_dir_1 = './data/' + dataset_name + '/result_1.csv'
test_dir_3 = './data/' + dataset_name + '/result_3.csv'
test_dir_5 = './data/' + dataset_name + '/result_5.csv'
feature_length = 1433
label_length = 7
if dataset_name == 'cora':
    dataset = dgl.data.CoraGraphDataset()
    g = dataset[0]
elif dataset_name == 'citeseer':
    dataset = dgl.data.CiteseerGraphDataset()
    g = dataset[0]
elif dataset_name == 'pubmed':
    dataset = dgl.data.PubmedGraphDataset()
    g = dataset[0]
elif dataset_name == 'squirrel':
    dataset = dgl.data.SquirrelDataset()
    g = dataset[0]
elif dataset_name == 'chameleon':
    dataset = dgl.data.ChameleonDataset()
    g = dataset[0]
elif dataset_name == 'actor':
    dataset = dgl.data.ActorDataset()
    g = dataset[0]
elif dataset_name == 'computer':
    dataset = dgl.data.AmazonCoBuyComputerDataset()
    g = dataset[0]
elif dataset_name == 'photo':
    dataset = dgl.data.AmazonCoBuyPhotoDataset()
    g = dataset[0]
else:
    raise ValueError('Dataset not available')

nodes_num = g.num_nodes()



class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        assert X.shape[0] == len(top_k_list)
        probs = np.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            all_labels.append(labels)
        return all_labels


def read_node_class():
    nodes_list = list()
    node_class_dict = dict()
    test_nodes_list = list()
    node2label_dict = dict()
    masks = torch.load(f'../data_mask/{dataset_name}_{t}_.pth')
    node_id = np.arange(0, nodes_num)
    for i in range(nodes_num):
        nodes_list.append(str(i))
    idx_train = node_id[masks['train_mask']]
    idx_val = node_id[masks['val_mask']]
    idx_test = node_id[masks['test_mask']]
    test_nodes_list = [str(i) for i in idx_test]
    label = g.ndata["label"].numpy()

    for i in range(nodes_num):
        node_class_dict[str(i)] = list()
        node_class_dict[str(i)].append(label[i])

    for n1 in node_class_dict.keys():
        label = [0.] * label_length
        for col in node_class_dict[n1]:
            label[int(col)] = 1.0
        node2label_dict[n1] = label
    nodes4train_list = list()
    for n1 in nodes_list:
        if n1 not in test_nodes_list:
            nodes4train_list.append(n1)
    return nodes4train_list, test_nodes_list, node2label_dict, node_class_dict


def read_embeddings(train_dir, test_dir_1, test_dir_3, test_dir_5):
    train_emb_dict = dict()
    feat = g.ndata["feat"].numpy()
    for i in range(g.num_nodes()):
        train_emb_dict[str(i)] = list(map(str, feat[i]))
    test_emb_1_dict = dict()
    with open(test_dir_1, "r") as f:
        lines = f.readlines()
        for line in lines:
            temp = list(line.strip('\n').split(' '))
            test_emb_1_dict[temp[0]] = temp[1:]
    test_emb_3_dict = dict()
    with open(test_dir_3, "r") as f:
        lines = f.readlines()
        for line in lines:
            temp = list(line.strip('\n').split(' '))
            test_emb_3_dict[temp[0]] = temp[1:]
    test_emb_5_dict = dict()
    with open(test_dir_5, "r") as f:
        lines = f.readlines()
        for line in lines:
            temp = list(line.strip('\n').split(' '))
            test_emb_5_dict[temp[0]] = temp[1:]
    return train_emb_dict, test_emb_1_dict, test_emb_3_dict, test_emb_5_dict


if __name__ == '__main__':
    nodes4train, test_nodes, node2label, node_class = read_node_class()
    train_emb, test_emb_1, test_emb_3, test_emb_5 = read_embeddings(train_dir, test_dir_1, test_dir_3, test_dir_5)
    all_results = defaultdict(list)
    all_results_1 = defaultdict(list)
    all_results_3 = defaultdict(list)
    all_results_5 = defaultdict(list)
    num_splits = 10
    for s in range(num_splits):
        masks = torch.load(f'../data_mask/{dataset_name}_{t}_.pth')
        node_id = np.arange(0, nodes_num)
        idx_train = node_id[masks['train_mask']]
        train_nodes = [str(i) for i in idx_train]
        
        X_train_, y_train_ = [], []
        for n in train_nodes:
            X_train_.append(train_emb[n])
            y_train_.append(node2label[n])
        X_test_, X_test_1, X_test_3, X_test_5, y_test_, y_test__ = [], [], [], [], [], []
        for n in test_nodes:
            X_test_.append(train_emb[n])
            X_test_1.append(test_emb_1[n])
            X_test_3.append(test_emb_3[n])
            X_test_5.append(test_emb_5[n])
            y_test_.append(node2label[n])
            y_test__.append(node_class[n])
        X_train = np.asarray(X_train_).astype(float)
        y_train = sparse.csr_matrix(np.asarray(y_train_).astype(int))
        X_test = np.asarray(X_test_).astype(float)
        X_test1 = np.asarray(X_test_1).astype(float)
        X_test3 = np.asarray(X_test_3).astype(float)
        X_test5 = np.asarray(X_test_5).astype(float)
        y_test = sparse.csr_matrix(np.asarray(y_test_).astype(int))

        clf = TopKRanker(LogisticRegression(solver='liblinear'))
        clf.fit(X_train, y_train)

        # find out how many labels should be predicted, same as deepwalk
        top_k_list = [len(l) for l in y_test__]
        preds = clf.predict(X_test, top_k_list)
        preds1 = clf.predict(X_test1, top_k_list)
        preds3 = clf.predict(X_test3, top_k_list)
        preds5 = clf.predict(X_test5, top_k_list)
        mlb = MultiLabelBinarizer(range(label_length))

        results = {}
        averages = ["micro", "macro"]
        for average in averages:
            results[average] = f1_score(mlb.fit_transform(y_test__), mlb.fit_transform(preds), average=average)
        results["accuracy"] =  accuracy_score(mlb.fit_transform(y_test__),  mlb.fit_transform(preds))
        all_results[s].append(results)
        results1 = {}
        averages = ["micro", "macro"]
        for average in averages:
            results1[average] = f1_score(mlb.fit_transform(y_test__), mlb.fit_transform(preds1), average=average)
        results1["accuracy"] =  accuracy_score(mlb.fit_transform(y_test__),  mlb.fit_transform(preds1))
        all_results_1[s].append(results1)
        results3 = {}
        averages = ["micro", "macro"]
        for average in averages:
            results3[average] = f1_score(mlb.fit_transform(y_test__), mlb.fit_transform(preds3), average=average)
        results3["accuracy"] =  accuracy_score(mlb.fit_transform(y_test__),  mlb.fit_transform(preds3))
        all_results_3[s].append(results3)
        results5 = {}
        averages = ["micro", "macro"]
        for average in averages:
            results5[average] = f1_score(mlb.fit_transform(y_test__), mlb.fit_transform(preds5), average=average)
        results5["accuracy"] =  accuracy_score(mlb.fit_transform(y_test__),  mlb.fit_transform(preds5))
        all_results_5[s].append(results5)

    print('---------------Results------------------')
    avg_score = defaultdict(float)
    for s in all_results.keys():
        for score_dict in all_results[s]:
            for metric, score in score_dict.items():
                avg_score[metric] += score
    for metric in avg_score:
        avg_score[metric] /= len(all_results)
    print(dict(avg_score))
    avg_score = defaultdict(float)
    for s in all_results_1.keys():
        for score_dict in all_results_1[s]:
            for metric, score in score_dict.items():
                avg_score[metric] += score
    for metric in avg_score:
        avg_score[metric] /= len(all_results_1)
    print(dict(avg_score))
    avg_score = defaultdict(float)
    for s in all_results_3.keys():
        for score_dict in all_results_3[s]:
            for metric, score in score_dict.items():
                avg_score[metric] += score
    for metric in avg_score:
        avg_score[metric] /= len(all_results_3)
    print(dict(avg_score))
    avg_score = defaultdict(float)
    for s in all_results_5.keys():
        for score_dict in all_results_5[s]:
            for metric, score in score_dict.items():
                avg_score[metric] += score
    for metric in avg_score:
        avg_score[metric] /= len(all_results_5)
    print(dict(avg_score))
    print('-------------------')
