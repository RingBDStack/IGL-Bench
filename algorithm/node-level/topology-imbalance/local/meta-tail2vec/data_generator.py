import dgl
import numpy as np
import os
import random
import tensorflow as tf
import torch
import tqdm
import math
import csv


def remove_blank_lines(input_file_path, output_file_path):
    with open(input_file_path, 'r', newline='') as csvfile_in, \
            open(output_file_path, 'w', newline='') as csvfile_out:
        reader = csv.reader(csvfile_in)
        writer = csv.writer(csvfile_out)

        wrote_row = False

        for row in reader:
            if any(field.strip() for field in row):
                writer.writerow(row)
                wrote_row = True
            elif wrote_row:
                wrote_row = False
    print("finish", output_file_path)


def cal_embedding(travel, hop_num_list, p_lambda, deepwalk_embeddings):
    weights = np.ndarray(shape=(len(travel),))
    index = 0
    for j in range(len(hop_num_list)):
        for k in range(hop_num_list[j]):
            weights[index] = math.exp(p_lambda * j)
            index += 1
    norm_weights = weights / weights.sum()
    index = 0
    temp_embeddings = np.zeros(shape=(len(travel), len(deepwalk_embeddings['0'])))
    for node in travel:
        temp_embeddings[index] = np.array(deepwalk_embeddings[node]).astype(np.float)
        index += 1
    embeddings = np.sum(np.multiply(temp_embeddings, norm_weights.reshape((-1, 1))), axis=0)
    return embeddings.tolist()


def generate_support_data(graph, source, deepwalk_embeddings, hop, node_max_size, p_lambda):
    hop_num_list = []
    frontiers = {source}
    travel = [source]
    travel_set = {source}
    travel_hop = 1
    while travel_hop <= hop:
        nexts = set()
        node_size = node_max_size[travel_hop - 1]
        for frontier in frontiers:
            if len(graph[frontier]) > node_size:
                node_children = np.random.choice(graph[frontier], node_size, replace=False)
            else:
                node_children = graph[frontier]
            for current in node_children:
                if current not in travel_set:
                    travel.append(current)
                    nexts.add(current)
                    travel_set.add(current)
        frontiers = nexts
        hop_num_list.append(len(nexts))
        travel_hop += 1
    travel.remove(source)
    feature_embedding = cal_embedding(travel, hop_num_list, p_lambda, deepwalk_embeddings)
    return deepwalk_embeddings[source], feature_embedding


def generate_query_data(graph, source, deepwalk_embeddings, s_n, hop, node_max_size, p_lambda):
    hop_num_list = []
    frontiers = {source}
    travel = [source]
    travel_set = {source}
    travel_hop = 1
    while travel_hop <= hop:
        nexts = set()
        node_size = node_max_size[travel_hop - 1]
        for frontier in frontiers:
            if travel_hop == 1:
                node_children = s_n
            else:
                if len(graph[frontier]) > node_size:
                    node_children = np.random.choice(list(graph[frontier]), node_size, replace=False)
                else:
                    node_children = graph[frontier]
            for current in node_children:
                if current not in travel_set:
                    travel.append(current)
                    nexts.add(current)
                    travel_set.add(current)
        frontiers = nexts
        hop_num_list.append(len(nexts))
        travel_hop += 1
    travel.remove(source)
    feature_embedding = cal_embedding(travel, hop_num_list, p_lambda, deepwalk_embeddings)
    return deepwalk_embeddings[source], feature_embedding


def write_task_to_file(s_n, q_n, g, emb, hop, size, p_lambda):
    task_data = []
    blank_row = [0.] * (2*len(emb['0'])+1)
    s_index = 0
    for n in s_n:
        oracle_embedding, embedding = generate_support_data(g, n, emb, hop=hop, node_max_size=size, p_lambda=p_lambda)
        task_data.append(list(n.split()) + oracle_embedding + embedding)
        s_index += 1
    while s_index < 5:
        task_data.append(blank_row)
        s_index += 1
    for n in q_n:
        oracle_embedding, embedding = generate_query_data(g, n, emb, s_n, hop=hop, node_max_size=size,
                                                          p_lambda=p_lambda)
        task_data.append(list(n.split()) + oracle_embedding + embedding)
    return task_data


class DataGenerator:
    def __init__(self, main_dir, dataset_name, kshot, meta_batchsz, t, total_batch_num=200):
        self.main_dir = main_dir
        self.kshot = kshot
        self.meta_batchsz = meta_batchsz
        self.total_batch_num = total_batch_num
        self.dataset_name = dataset_name
        self.hop = 2
        self.size1 = 50
        self.size2 = 25
        self.p_lambda = 0
        self.t = t

        self.metatrain_file = self.main_dir + dataset_name + '/train.csv'
        self.metatest_file = self.main_dir + dataset_name + '/test.csv'

        self.graph_dir = self.main_dir + dataset_name + '/graph.adjlist'
        self.graph_dense_dir = self.main_dir + dataset_name + '/graph_dense.adjlist'
        self.emb_dir = self.main_dir + dataset_name + '/embs.pth'

        print(dataset_name)
        if dataset_name == 'cora':
            dataset = dgl.data.CoraGraphDataset()
            g = dataset[0]
        elif dataset_name == 'citeseer':
            dataset = dgl.data.CiteseerGraphDataset()
            g = dataset[0]
        elif dataset_name == 'pubmed':
            dataset = dgl.data.PubmedGraphDataset()
            g = dataset[0]
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
                    feature_list[int(index)-1] = 1
                feature.append(feature_list)
                labels.append(label)

            features = np.array(feature)
            label_raw = np.array(labels)

            nodes = np.arange(0, len(features))
        elif dataset_name == 'computer':
            dataset = dgl.data.AmazonCoBuyComputerDataset()
            g = dataset[0]
        elif dataset_name == 'photo':
            dataset = dgl.data.AmazonCoBuyPhotoDataset()
            g = dataset[0]
        elif dataset_name == 'arxiv':
            from ogb.nodeproppred import DglNodePropPredDataset
            dataset = DglNodePropPredDataset(name='ogbn-arxiv')
            g = dataset[0][0]
            g.ndata['label'] = dataset[0][1].squeeze()
            from sklearn.preprocessing import StandardScaler
            norm = StandardScaler()
            norm.fit(g.ndata['feat'])
            g.ndata['feat'] = torch.tensor(norm.transform(g.ndata['feat'])).float()
        else:
            raise ValueError('Dataset not available')

        
        if dataset_name == 'cora' or dataset_name == 'citeseer' or dataset_name == 'pubmed' or dataset_name == 'computer' or dataset_name == 'photo' or dataset_name == 'arxiv':
            self.nodes_num = g.num_nodes()

            self.graph = dict()
            src, dst = g.edges()
            src = src.numpy()
            dst = dst.numpy()
            for i in range(self.nodes_num):
                self.graph[str(i)] = list()
            for (s, d) in zip(src, dst):
                self.graph[str(s)].append(str(d))

            sparse_node_set = set()
            for i in self.graph.keys():
                if len(self.graph[i]) <= 5:
                    sparse_node_set.add(i)

            node_type = node_type_info(self.graph, sparse_node_set)

            self.graph_dense = dict()
            for key in self.graph.keys():
                self.graph_dense[key] = set()
                for item in self.graph[key]:
                    if node_type[item] == 'dense':
                        self.graph_dense[key].add(str(item))

            self.deepwalk_emb = dict()
            feat = g.ndata["feat"].numpy()
            for i in range(g.num_nodes()):
                self.deepwalk_emb[str(i)] = list(map(str, feat[i]))

            # temp = torch.load(self.main_dir + dataset_name + '/embs.pth')
            # for i in range(self.nodes_num):
            #     self.deepwalk_emb[str(i)] = list(map(str, temp[i]))


            self.emb_dim = len(self.deepwalk_emb['0'])
        
        else:
            self.nodes_num = len(features)

            self.graph = dict()
            for i in range(self.nodes_num):
                self.graph[str(i)] = list()
            for (s, d) in zip(src, dst):
                self.graph[str(s)].append(str(d))

            sparse_node_set = set()
            for i in self.graph.keys():
                if len(self.graph[i]) <= 5:
                    sparse_node_set.add(i)

            node_type = node_type_info(self.graph, sparse_node_set)

            self.graph_dense = dict()
            for key in self.graph.keys():
                self.graph_dense[key] = set()
                for item in self.graph[key]:
                    if node_type[item] == 'dense':
                        self.graph_dense[key].add(str(item))

            self.deepwalk_emb = dict()
            feat = features
            for i in range(self.nodes_num):
                self.deepwalk_emb[str(i)] = list(map(str, feat[i]))

            # temp = torch.load(self.main_dir + dataset_name + '/embs.pth')
            # for i in range(self.nodes_num):
            #     self.deepwalk_emb[str(i)] = list(map(str, temp[i]))

            self.emb_dim = len(self.deepwalk_emb['0'])

        print("finish")


    def make_data_tensor(self, training=True):
        num_total_batches = self.total_batch_num
        if training:
            file = self.metatrain_file
        else:
            file = self.metatest_file

        if training:
            if os.path.exists('./data/' + self.dataset_name + f'/{self.t}_trainfile.csv'):
                pass
            else:
                all_data = []
                train_nodes = []
                masks = torch.load(f'../data_mask/{self.dataset_name}_{self.t}_.pth')
                node_id = np.arange(0, self.nodes_num)
                idx_train = node_id[masks['train_mask']]
                idx_val = node_id[masks['val_mask']]
                idx_test = node_id[masks['test_mask']]

                train_nodes = list(map(str, idx_train))

                for _ in tqdm.tqdm(range(num_total_batches), 'generating episodes'):
                    query_node = random.sample(train_nodes, 1)
                    # print(query_node)
                    if len(self.graph_dense[query_node[0]]) > self.kshot:
                        support_node = random.sample(self.graph_dense[query_node[0]], self.kshot)
                    else:
                        support_node = self.graph_dense[query_node[0]]
                    task_data = write_task_to_file(support_node, query_node, self.graph, self.deepwalk_emb, self.hop,
                                                   (self.size1, self.size2), self.p_lambda)
                    all_data.extend(task_data)

                with open('./data/' + self.dataset_name + f'/{self.t}_trainfile_raw.csv', 'w') as fw:
                    writer = csv.writer(fw)
                    writer.writerows(all_data)
                    print('save train file list to trainfile.csv')
                remove_blank_lines('./data/' + self.dataset_name + f'/{self.t}_trainfile_raw.csv',
                                   './data/' + self.dataset_name + f'/{self.t}_trainfile.csv')
        else:
            if os.path.exists('./data/' + self.dataset_name + f'/{self.t}_testfile.csv'):
                pass
            else:
                all_data = []
                test_nodes = []
                other_nodes = []
                masks = torch.load(f'../data_mask/{self.dataset_name}_{self.t}_.pth')
                node_id = np.arange(0, self.nodes_num)
                idx_train = node_id[masks['train_mask']]
                idx_val = node_id[masks['val_mask']]
                idx_test = node_id[masks['test_mask']]

                test_nodes = list(map(str, idx_test))

                for n in tqdm.tqdm(test_nodes, 'generating test episodes'):
                    query_node = list()
                    query_node.append(n)
                    # print(query_node)
                    if len(self.graph_dense[query_node[0]]) > self.kshot:
                        support_node = random.sample(self.graph_dense[query_node[0]], self.kshot)
                    else:
                        support_node = self.graph_dense[query_node[0]]
                    task_data = write_task_to_file(support_node, query_node, self.graph, self.deepwalk_emb,
                                                   self.hop, (self.size1, self.size2), self.p_lambda)
                    all_data.extend(task_data)
                with open('./data/' + self.dataset_name + f'/{self.t}_testfile_raw.csv', 'w') as fw:
                    writer = csv.writer(fw)
                    writer.writerows(all_data)
                    print('save test file list to testfile.csv')
                remove_blank_lines('./data/' + self.dataset_name + f'/{self.t}_testfile_raw.csv',
                                   './data/' + self.dataset_name + f'/{self.t}_testfile.csv')


        print('creating pipeline ops')
        if training:
            filename_queue = tf.train.string_input_producer(['./data/' + self.dataset_name + f'/{self.t}_trainfile.csv'],
                                                            shuffle=False)
        else:
            filename_queue = tf.train.string_input_producer(['./data/' + self.dataset_name + f'/{self.t}_testfile.csv'],
                                                            shuffle=False)
        reader = tf.TextLineReader()
        _, value = reader.read(filename_queue)
        record_defaults = [0.] * (2*self.emb_dim+1)
        row = tf.decode_csv(value, record_defaults=record_defaults)
        feature_and_label = tf.stack(row)

        print('batching data')
        examples_per_batch = 1 + self.kshot
        batch_data_size = self.meta_batchsz * examples_per_batch
        features = tf.train.batch(
            [feature_and_label],
            batch_size=batch_data_size,
            num_threads=1,
            capacity=256,
        )
        all_node_id = []
        all_label_batch = []
        all_feature_batch = []
        for i in range(self.meta_batchsz):
            data_batch = features[i * examples_per_batch:(i + 1) * examples_per_batch]
            node_id, label_batch, feature_batch = tf.split(data_batch, [1, self.emb_dim, self.emb_dim], axis=1)
            all_node_id.append(node_id)
            all_label_batch.append(label_batch)
            all_feature_batch.append(feature_batch)
        all_node_id = tf.stack(all_node_id)
        all_label_batch = tf.stack(all_label_batch)
        all_feature_batch = tf.stack(all_feature_batch)
        return all_node_id, all_label_batch, all_feature_batch


def node_type_info(graph_dict, sparse_node_set):
    node_type_dict = dict()
    for n in graph_dict.keys():
        if n in sparse_node_set:
            node_type_dict[n] = 'sparse'
        else:
            neighbor = len(graph_dict[n])
            for adj in graph_dict[n]:
                if adj in sparse_node_set:
                    neighbor -= 1
            if neighbor > 5:
                node_type_dict[n] = 'dense'
            else:
                node_type_dict[n] = 'middle'
    s_num = 0
    d_num = 0
    m_num = 0
    for n in node_type_dict.keys():
        if node_type_dict[n] == 'sparse':
            s_num += 1
        if node_type_dict[n] == 'dense':
            d_num += 1
        if node_type_dict[n] == 'middle':
            m_num += 1
    return node_type_dict