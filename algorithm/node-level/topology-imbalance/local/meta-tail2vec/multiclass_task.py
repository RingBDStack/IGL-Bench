import numpy as np
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from collections import defaultdict
from sklearn.metrics import accuracy_score

dataset = ''
train_dir = './data/' + dataset + '/graph.embeddings'
test_dir_1 = './data/' + dataset + '/result_1.csv'
test_dir_3 = './data/' + dataset + '/result_3.csv'
test_dir_5 = './data/' + dataset + '/result_5.csv'


def read_node_class():
    nodes_list = list()
    node_class_dict = dict()
    test_nodes_list = list()
    class_num_dict = dict()
    with open('./data/' + dataset + '/test.csv', "r") as f:
        lines = f.readlines()
        for line in lines:
            temp = list(line.strip('\n').split(' '))
            test_nodes_list.append(temp[0])
    with open('./data/' + dataset + '/node_class.txt', "r") as f:
        lines = f.readlines()
        for line in lines:
            temp = list(line.strip('\n').split(' '))
            if temp[1] not in class_num_dict.keys():
                class_num_dict[temp[1]] = 1
            else:
                class_num_dict[temp[1]] += 1
            nodes_list.append(temp[0])
            node_class_dict[temp[0]] = temp[1]
    nodes4train_list = list()
    for n in nodes_list:
        if n not in test_nodes_list:
            nodes4train_list.append(n)
    return nodes4train_list, test_nodes_list, node_class_dict, class_num_dict


def read_embeddings(train_dir, test_dir_1, test_dir_3, test_dir_5):
    train_emb_dict = dict()
    with open(train_dir, "r") as f:
        lines = f.readlines()
        for line in lines:
            temp = list(line.strip('\n').split(' '))
            if len(temp) == 2:
                continue
            else:
                train_emb_dict[temp[0]] = temp[1:]
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
    nodes4train, test_nodes, node_class, class_num = read_node_class()
    max_class = max(class_num, key = class_num.get)
    microf_labels = list(class_num.keys())
    microf_labels.remove(str(max_class))
    microf_labels = [int(x) for x in microf_labels]
    train_emb, test_emb_1, test_emb_3, test_emb_5 = read_embeddings(train_dir, test_dir_1, test_dir_3, test_dir_5)
    all_results = defaultdict(list)
    all_results_1 = defaultdict(list)
    all_results_3 = defaultdict(list)
    all_results_5 = defaultdict(list)
    num_splits = 10
    for s in range(num_splits):
        train_nodes, _, _, _ = train_test_split(nodes4train, range(len(nodes4train)), train_size=100,
                                                random_state=19 + s * 7)
        X_train_, y_train_ = [], []
        for n in train_nodes:
            X_train_.append(train_emb[n])
            y_train_.append(node_class[n])
        X_test_, X_test_1, X_test_3, X_test_5, y_test_, y_test__ = [], [], [], [], [], []
        for n in test_nodes:
            X_test_.append(train_emb[n])
            X_test_1.append(test_emb_1[n])
            X_test_3.append(test_emb_3[n])
            X_test_5.append(test_emb_5[n])
            y_test_.append(node_class[n])
        X_train = np.asarray(X_train_).astype(float)
        y_train = np.asarray(y_train_).astype(float)
        X_test = np.asarray(X_test_).astype(float)
        X_test1 = np.asarray(X_test_1).astype(float)
        X_test3 = np.asarray(X_test_3).astype(float)
        X_test5 = np.asarray(X_test_5).astype(float)
        y_test = np.asarray(y_test_).astype(float)

        clf = LogisticRegression(multi_class='auto', solver='liblinear')
        clf.fit(X_train, y_train)

        preds = clf.predict(X_test)
        preds1 = clf.predict(X_test1)
        preds3 = clf.predict(X_test3)
        preds5 = clf.predict(X_test5)

        results = {}
        averages = ["micro", "macro"]
        for average in averages:
            if average == "micro":
                results[average] = f1_score(y_test, preds, average=average, labels=np.asarray(microf_labels))
            else:
                results[average] = f1_score(y_test, preds, average=average)
        results["accuracy"] =  accuracy_score(y_test, preds)
        all_results[s].append(results)
        results1 = {}
        averages = ["micro", "macro"]
        for average in averages:
            if average == "micro":
                results1[average] = f1_score(y_test, preds1, average=average, labels=np.asarray(microf_labels))
            else:
                results1[average] = f1_score(y_test, preds1, average=average)
        results1["accuracy"] =  accuracy_score(y_test, preds1)
        all_results_1[s].append(results1)
        results3 = {}
        averages = ["micro", "macro"]
        for average in averages:
            if average == "micro":
                results3[average] = f1_score(y_test, preds3, average=average, labels=np.asarray(microf_labels))
            else:
                results3[average] = f1_score(y_test, preds3, average=average)
        results3["accuracy"] =  accuracy_score(y_test, preds3)
        all_results_3[s].append(results3)
        results5 = {}
        averages = ["micro", "macro"]
        for average in averages:
            if average == "micro":
                results5[average] = f1_score(y_test, preds5, average=average, labels=np.asarray(microf_labels))
            else:
                results5[average] = f1_score(y_test, preds5, average=average)
        results5["accuracy"] =  accuracy_score(y_test, preds5)
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
