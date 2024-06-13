from sklearn.model_selection import train_test_split

dataset = 'cora'
sparse_node_threshold = 5


def read_flickr_graph():
    node_dict = dict()
    graph_dict = dict()
    class_dict = dict()
    node_class = dict()
    node_index = 1
    with open('./dataset/' + dataset + '/nodes.csv', "r") as fr:
        lines = fr.readlines()
        for line in lines:
            temp = list(line.strip('\n').split(' '))
            node_dict[temp[0]] = node_index
            node_index += 1
    with open('./dataset/' + dataset + '/nodes.csv', "r") as fr:
        lines = fr.readlines()
        for line in lines:
            temp = list(line.strip('\n').split(' '))
            if node_dict[temp[0]] not in graph_dict.keys():
                graph_dict[node_dict[temp[0]]] = set()
    edge_num = 0
    with open('./dataset/' + dataset + '/edges.csv', "r") as fr:
        lines = fr.readlines()
        for line in lines:
            edge_num += 1
            temp = list(line.strip('\n').split(','))
            graph_dict[node_dict[temp[0]]].add(node_dict[temp[1]])
            graph_dict[node_dict[temp[1]]].add(node_dict[temp[0]])
    class_index = 0
    with open('./dataset/' + dataset + '/groups.csv', "r") as fr:
        lines = fr.readlines()
        for line in lines:
            temp = list(line.strip('\n').split(' '))
            if temp[0] not in class_dict.keys():
                class_dict[temp[0]] = class_index
                class_index += 1
    with open('./dataset/' + dataset + '/group-edges.csv', "r") as fr:
        lines = fr.readlines()
        for line in lines:
            temp = list(line.strip('\n').split(','))
            if node_dict[temp[0]] not in node_class.keys():
                node_class[node_dict[temp[0]]] = set()
                node_class[node_dict[temp[0]]].add(class_dict[temp[1]])
            else:
                node_class[node_dict[temp[0]]].add(class_dict[temp[1]])
    print('dataset ' + dataset)
    print('total node number: %d' % len(graph_dict.keys()))
    print('total edge number: %d' % edge_num)
    print('total class number: %d' % len(class_dict.keys()))
    return node_dict, graph_dict, node_class, class_dict


def read_email_graph():
    node_dict = dict()
    graph_dict = dict()
    node_class = dict()
    classset = set()
    node_index = 1
    with open('./dataset/' + dataset + '/cora_edgelist.txt', "r") as fr:
        lines = fr.readlines()
        for line in lines:
            temp = list(line.strip('\n').split(' '))
            if temp[0] not in node_dict.keys():
                node_dict[temp[0]] = node_index
                node_index += 1
            if temp[1] not in node_dict.keys():
                node_dict[temp[1]] = node_index
                node_index += 1
    # reverse_node_dict = dict(zip(node_dict.values(), node_dict.keys()))
    # with open('./dataset/' + dataset + '/node_dict.txt', "w") as fw:
    #     for n in node_dict:
    #         fw.write(str(n) + ' ' + str(node_dict) + '\n')
    for n in node_dict.keys():
        graph_dict[node_dict[n]] = set()
    edge_num = 0
    with open('./dataset/' + dataset + '/cora_edgelist.txt', "r") as fr:
        lines = fr.readlines()
        for line in lines:
            edge_num += 1
            temp = list(line.strip('\n').split(' '))
            graph_dict[node_dict[temp[0]]].add(node_dict[temp[1]])
            graph_dict[node_dict[temp[1]]].add(node_dict[temp[0]])
    with open('./dataset/' + dataset + '/cora_labels.txt', "r") as fr:
        lines = fr.readlines()
        for line in lines:
            temp = list(line.strip('\n').split(' '))
            if temp[0] in node_dict.keys():
                node_class[node_dict[temp[0]]] = temp[1]
                classset.add(temp[1])
    print(len(classset))
    print('dataset ' + dataset)
    print('total node number: %d' % len(graph_dict.keys()))
    print('total edge number: %d' % edge_num)
    return node_dict, graph_dict, node_class


def read_wiki_graph():
    node_dict = dict()
    graph_dict = dict()
    node_class = dict()
    node_index = 1
    with open('./dataset/' + dataset + '/Wiki_edgelist.txt', "r") as fr:
        lines = fr.readlines()
        for line in lines:
            temp = list(line.strip('\n').split('\t'))
            if temp[0] not in node_dict.keys():
                node_dict[temp[0]] = node_index
                node_index += 1
            if temp[1] not in node_dict.keys():
                node_dict[temp[1]] = node_index
                node_index += 1
    # reverse_node_dict = dict(zip(node_dict.values(), node_dict.keys()))
    # with open('./dataset/' + dataset + '/node_dict.txt', "w") as fw:
    #     for n in node_dict:
    #         fw.write(str(n) + ' ' + str(node_dict) + '\n')
    for n in node_dict.keys():
        graph_dict[node_dict[n]] = set()
    edge_num = 0
    with open('./dataset/' + dataset + '/Wiki_edgelist.txt', "r") as fr:
        lines = fr.readlines()
        for line in lines:
            edge_num += 1
            temp = list(line.strip('\n').split('\t'))
            graph_dict[node_dict[temp[0]]].add(node_dict[temp[1]])
            graph_dict[node_dict[temp[1]]].add(node_dict[temp[0]])
    with open('./dataset/' + dataset + '/Wiki_category.txt', "r") as fr:
        lines = fr.readlines()
        for line in lines:
            temp = list(line.strip('\n').split('\t'))
            if temp[0] in node_dict.keys():
                node_class[node_dict[temp[0]]] = temp[1]
    print('dataset ' + dataset)
    print('total node number: %d' % len(graph_dict.keys()))
    print('total edge number: %d' % edge_num)
    return node_dict, graph_dict, node_class


def select_sparse_node(graph_dict):
    sparse_node_set = set()
    for i in graph_dict.keys():
        if len(graph_dict[i]) <= sparse_node_threshold:
            sparse_node_set.add(i)
    return sparse_node_set


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
            if neighbor > sparse_node_threshold:
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


def generate_train_test_candidate_flickr(graph_dict, sparse_node_set, node_type_dict):
    train_node_candidate_set = set()
    test_node_candidate_set_all = set()
    test_node_candidate_set_1 = set()
    test_node_candidate_set_2 = set()
    test_node_candidate_set_3 = set()
    test_node_candidate_set_4 = set()
    test_node_candidate_set_5 = set()
    test_nodes_list = list()
    test_node_candidate_num = 0
    for n in sparse_node_set:
        dense_num = 0
        middle_num = 0
        sparse_num = 0
        for adj in graph_dict[n]:
            if node_type_dict[adj] == 'dense':
                dense_num += 1
            elif node_type_dict[adj] == 'middle':
                middle_num += 1
            else:
                sparse_num += 1
        if middle_num == 0 and sparse_num == 0:
            test_node_candidate_set_all.add(n)
            if len(graph_dict[n]) == 1:
                test_node_candidate_set_1.add(n)
            if len(graph_dict[n]) == 2:
                test_node_candidate_set_2.add(n)
            if len(graph_dict[n]) == 3:
                test_node_candidate_set_3.add(n)
            if len(graph_dict[n]) == 4:
                test_node_candidate_set_4.add(n)
            if len(graph_dict[n]) == 5:
                test_node_candidate_set_5.add(n)
            test_node_candidate_num += 1
    print(len(test_node_candidate_set_1))
    print(len(test_node_candidate_set_2))
    print(len(test_node_candidate_set_3))
    print(len(test_node_candidate_set_4))
    print(len(test_node_candidate_set_5))
    print(test_node_candidate_num)
    for n in graph_dict.keys():
        if n in sparse_node_set:
            continue
        else:
            if node_type_dict[n] == 'middle':
                continue
            else:
                neighbor = len(graph_dict[n])
                for adj in graph_dict[n]:
                    if adj in sparse_node_set:
                        neighbor -= 1
                if 36 < neighbor <= 56:
                    train_node_candidate_set.add(n)
    # flickr do not have 1 shot node
    # test_nodes_list_1, _, _, _ = train_test_split(list(test_node_candidate_set_1),
    #                                               range(len(test_node_candidate_set_1)), train_size=157,
    #                                               random_state=19)
    # test_nodes_list.extend(test_nodes_list_1)
    test_nodes_list_2, _, _, _ = train_test_split(list(test_node_candidate_set_2),
                                                  range(len(test_node_candidate_set_2)), train_size=320,
                                                  random_state=19)
    test_nodes_list.extend(test_nodes_list_2)
    test_nodes_list_3, _, _, _ = train_test_split(list(test_node_candidate_set_3),
                                                  range(len(test_node_candidate_set_3)), train_size=260,
                                                  random_state=19)
    test_nodes_list.extend(test_nodes_list_3)
    test_nodes_list_4, _, _, _ = train_test_split(list(test_node_candidate_set_4),
                                                  range(len(test_node_candidate_set_4)), train_size=227,
                                                  random_state=19)
    test_nodes_list.extend(test_nodes_list_4)
    test_nodes_list_5, _, _, _ = train_test_split(list(test_node_candidate_set_5),
                                                  range(len(test_node_candidate_set_5)), train_size=193,
                                                  random_state=19)
    test_nodes_list.extend(test_nodes_list_5)
    test_node_candidate_set = set(test_node_candidate_set_all)
    return train_node_candidate_set, test_node_candidate_set


def generate_train_test_candidate_email(graph_dict, sparse_node_set, node_type_dict):
    test_node_candidate_num = 0
    test_node_candidate_set_all = set()
    train_node_candidate_set = set()
    test_node_candidate_set_1 = set()
    test_node_candidate_set_2 = set()
    test_node_candidate_set_3 = set()
    test_node_candidate_set_4 = set()
    test_node_candidate_set_5 = set()
    test_nodes_list = list()
    for n in sparse_node_set:
        dense_num = 0
        middle_num = 0
        sparse_num = 0
        for adj in graph_dict[n]:
            if node_type_dict[adj] == 'dense':
                dense_num += 1
            elif node_type_dict[adj] == 'middle':
                middle_num += 1
            else:
                sparse_num += 1
        if middle_num == 0 and sparse_num == 0:
            test_node_candidate_set_all.add(n)
            if len(graph_dict[n]) == 1:
                test_node_candidate_set_1.add(n)
            if len(graph_dict[n]) == 2:
                test_node_candidate_set_2.add(n)
            if len(graph_dict[n]) == 3:
                test_node_candidate_set_3.add(n)
            if len(graph_dict[n]) == 4:
                test_node_candidate_set_4.add(n)
            if len(graph_dict[n]) == 5:
                test_node_candidate_set_5.add(n)
            test_node_candidate_num += 1
    print(test_node_candidate_num)
    print(len(test_node_candidate_set_1))
    print(len(test_node_candidate_set_2))
    print(len(test_node_candidate_set_3))
    print(len(test_node_candidate_set_4))
    print(len(test_node_candidate_set_5))
    for n in graph_dict.keys():
        if n in sparse_node_set:
            continue
        else:
            if node_type_dict[n] == 'middle':
                continue
            else:
                neighbor = len(graph_dict[n])
                for adj in graph_dict[n]:
                    if adj in sparse_node_set:
                        neighbor -= 1
                if 12 < neighbor <= 32:
                    train_node_candidate_set.add(n)
    test_nodes_list_1, _, _, _ = train_test_split(list(test_node_candidate_set_1),
                                                  range(len(test_node_candidate_set_1)), train_size=43,
                                                  random_state=19)
    test_nodes_list.extend(test_nodes_list_1)
    test_nodes_list_2, _, _, _ = train_test_split(list(test_node_candidate_set_2),
                                                  range(len(test_node_candidate_set_2)), train_size=17,
                                                  random_state=19)
    test_nodes_list.extend(test_nodes_list_2)
    test_nodes_list_3, _, _, _ = train_test_split(list(test_node_candidate_set_3),
                                                  range(len(test_node_candidate_set_3)), train_size=15,
                                                  random_state=19)
    test_nodes_list.extend(test_nodes_list_3)
    test_nodes_list_4, _, _, _ = train_test_split(list(test_node_candidate_set_4),
                                                  range(len(test_node_candidate_set_4)), train_size=11,
                                                  random_state=19)
    test_nodes_list.extend(test_nodes_list_4)
    test_nodes_list_5, _, _, _ = train_test_split(list(test_node_candidate_set_5),
                                                  range(len(test_node_candidate_set_5)), train_size=14,
                                                  random_state=19)
    test_nodes_list.extend(test_nodes_list_5)
    test_node_candidate_set = set(test_nodes_list)
    print('train node candidate num is %d' % len(train_node_candidate_set))
    print('test node candidate num is %d' % len(test_node_candidate_set))
    print('=================================')
    return train_node_candidate_set, test_node_candidate_set


def generate_train_test_candidate_wiki(graph_dict, sparse_node_set, node_type_dict):
    test_node_candidate_num = 0
    test_node_candidate_set_all = set()
    train_node_candidate_set = set()
    test_node_candidate_set_1 = set()
    test_node_candidate_set_2 = set()
    test_node_candidate_set_3 = set()
    test_node_candidate_set_4 = set()
    test_node_candidate_set_5 = set()
    test_nodes_list = list()
    for n in sparse_node_set:
        dense_num = 0
        middle_num = 0
        sparse_num = 0
        for adj in graph_dict[n]:
            if node_type_dict[adj] == 'dense':
                dense_num += 1
            elif node_type_dict[adj] == 'middle':
                middle_num += 1
            else:
                sparse_num += 1
        if middle_num == 0 and sparse_num == 0:
            test_node_candidate_set_all.add(n)
            if len(graph_dict[n]) == 1:
                test_node_candidate_set_1.add(n)
            if len(graph_dict[n]) == 2:
                test_node_candidate_set_2.add(n)
            if len(graph_dict[n]) == 3:
                test_node_candidate_set_3.add(n)
            if len(graph_dict[n]) == 4:
                test_node_candidate_set_4.add(n)
            if len(graph_dict[n]) == 5:
                test_node_candidate_set_5.add(n)
            test_node_candidate_num += 1
    print(test_node_candidate_num)
    print(len(test_node_candidate_set_1))
    print(len(test_node_candidate_set_2))
    print(len(test_node_candidate_set_3))
    print(len(test_node_candidate_set_4))
    print(len(test_node_candidate_set_5))
    for n in graph_dict.keys():
        if n in sparse_node_set:
            continue
        else:
            if node_type_dict[n] == 'middle':
                continue
            else:
                neighbor = len(graph_dict[n])
                for adj in graph_dict[n]:
                    if adj in sparse_node_set:
                        neighbor -= 1
                if 7 < neighbor <= 20:
                    train_node_candidate_set.add(n)
    test_nodes_list_1, _, _, _ = train_test_split(list(test_node_candidate_set_1),
                                                  range(len(test_node_candidate_set_1)), train_size=30,
                                                  random_state=19)
    test_nodes_list.extend(test_nodes_list_1)
    test_nodes_list_2, _, _, _ = train_test_split(list(test_node_candidate_set_2),
                                                  range(len(test_node_candidate_set_2)), train_size=24,
                                                  random_state=19)
    test_nodes_list.extend(test_nodes_list_2)
    test_nodes_list_3, _, _, _ = train_test_split(list(test_node_candidate_set_3),
                                                  range(len(test_node_candidate_set_3)), train_size=20,
                                                  random_state=19)
    test_nodes_list.extend(test_nodes_list_3)
    test_nodes_list_4, _, _, _ = train_test_split(list(test_node_candidate_set_4),
                                                  range(len(test_node_candidate_set_4)), train_size=13,
                                                  random_state=19)
    test_nodes_list.extend(test_nodes_list_4)
    test_nodes_list_5, _, _, _ = train_test_split(list(test_node_candidate_set_5),
                                                  range(len(test_node_candidate_set_5)), train_size=13,
                                                  random_state=19)
    test_nodes_list.extend(test_nodes_list_5)
    test_node_candidate_set = set(test_nodes_list)
    print('train node candidate num is %d' % len(train_node_candidate_set))
    print('test node candidate num is %d' % len(test_node_candidate_set))
    print('=================================')
    return train_node_candidate_set, test_node_candidate_set


def generate_train_test_set(train_node_candidate_set, test_node_candidate_set):
    train_node_set = train_node_candidate_set
    return train_node_set, test_node_candidate_set


def save_graph(graph_dict, graph_dir):
    with open(graph_dir, 'w') as fw:
        for key in graph_dict.keys():
            fw.write(str(key))
            for item in graph_dict[key]:
                fw.write(' ' + str(item))
            fw.write('\n')


def save_node(node_set, node_save_dir):
    with open(node_save_dir, 'w') as fw:
        for n in node_set:
            fw.write(str(n))
            fw.write('\n')


def save_dense_graph(graph_dict, graph_dir, node_type_dict):
    with open(graph_dir, 'w') as fw:
        for key in graph_dict.keys():
            fw.write(str(key))
            for item in graph_dict[key]:
                if node_type_dict[item] == 'dense':
                    fw.write(' ' + str(item))
            fw.write('\n')


def save_node_class_info(node_info, save_dir):
    with open(save_dir, 'w') as fw:
        for key in node_info.keys():
            fw.write(str(key))
            for item in node_info[key]:
                fw.write(' ' + str(item))
            fw.write('\n')


def train_flickr():
    node2id, graph, node_class_info, class2id = read_flickr_graph()
    sparse_node = select_sparse_node(graph)   #choose low degree nodes
    node_type = node_type_info(graph, sparse_node)    #get node class
    train_node_candidate, test_node_candidate = generate_train_test_candidate_flickr(graph, sparse_node, node_type)
    train_node, test_node = generate_train_test_set(train_node_candidate, test_node_candidate)
    save_node(train_node, './data/' + dataset + '/train.csv')
    save_node(test_node, './data/' + dataset + '/test.csv')   #set
    save_node_class_info(node_class_info, './data/' + dataset + '/node_class.txt')  #dict
    save_graph(graph, './data/' + dataset + '/graph.adjlist')   #全图
    save_dense_graph(graph, './data/' + dataset + '/graph_dense.adjlist', node_type) #终点是不是dense


def train_email():
    node2id, graph, node_class_info = read_email_graph()
    sparse_node = select_sparse_node(graph)
    node_type = node_type_info(graph, sparse_node)
    train_node_candidate, test_node_candidate = generate_train_test_candidate_email(graph, sparse_node, node_type)
    train_node, test_node = generate_train_test_set(train_node_candidate, test_node_candidate)
    save_node(train_node, './data/' + dataset + '/train.csv')
    save_node(test_node, './data/' + dataset + '/test.csv')
    save_node_class_info(node_class_info, './data/' + dataset + '/node_class.txt')
    save_graph(graph, './data/' + dataset + '/graph.adjlist')
    save_dense_graph(graph, './data/' + dataset + '/graph_dense.adjlist', node_type)


def train_wiki():
    node2id, graph, node_class_info = read_wiki_graph()
    sparse_node = select_sparse_node(graph)
    node_type = node_type_info(graph, sparse_node)
    train_node_candidate, test_node_candidate = generate_train_test_candidate_wiki(graph, sparse_node, node_type)
    train_node, test_node = generate_train_test_set(train_node_candidate, test_node_candidate)
    save_node(train_node, './data/' + dataset + '/train.csv')
    save_node(test_node, './data/' + dataset + '/test.csv')
    save_node_class_info(node_class_info, './data/' + dataset + '/node_class.txt')
    save_graph(graph, './data/' + dataset + '/graph.adjlist')
    save_dense_graph(graph, './data/' + dataset + '/graph_dense.adjlist', node_type)


if __name__ == '__main__':
    if dataset == 'flickr':
        train_flickr()
    elif dataset == 'email' or dataset == 'cora':
        train_email()
    else:
        train_wiki()
