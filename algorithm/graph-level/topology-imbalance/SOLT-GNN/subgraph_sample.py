import numpy as np
import argparse

from util import load_data


def subgraph_sample(dataset, graph_list, border, nums=500):
    print("Sampling Data")

    with open('dataset/%s/sampling.txt' % (dataset), 'w') as f:
        f.write(str(len(graph_list)) + '\n')
        for graph in graph_list:
            if graph.nodegroup == 1:
                graph.sample_list = []
                graph.unsample_list = []
                graph.sample_x = []
                n = graph.g.number_of_nodes()
                K = int(min(border - 1, n / 2))
                f.write(str(K) + '\n')
                graph.K = K
                for i in range(nums):
                    sample_idx = np.random.permutation(n)
                    j = 0
                    sample_set = set()
                    wait_set = []
                    cnt = 0
                    if (len(graph.neighbors[j]) == 0):
                        j += 1
                    wait_set.append(sample_idx[j])
                    while cnt < K:
                        if len(wait_set) != 0:
                            x = wait_set.pop()
                        else:
                            break
                        while x in sample_set:
                            if len(wait_set) != 0:
                                x = wait_set.pop()
                            else:
                                cnt = K
                                break
                        sample_set.add(x)
                        cnt += 1
                        wait_set.extend(graph.neighbors[x])
                    unsample_set = set(range(n)).difference(sample_set)
                    f.write(str(len(sample_set)) + ' ')
                    for x in list(sample_set):
                        f.write(str(x) + ' ')
                    for x in list(unsample_set):
                        f.write(str(x) + ' ')
                    f.write('\n')
            else:
                f.write('0\n')
    print("%s Finished" % (dataset))


if __name__ == '__main__':
    data_list = ['PTC_MR', 'FRANK', 'DD', 'PROTEINS', 'IMDB-BINARY', 'REDDIT-BINARY']
    for dataset in data_list:

        np.random.seed(0)

        graphs, num_classes = load_data(dataset, 0)

        if dataset == 'PTC':
            border = 22
        elif dataset == 'PTC_MR':
            border = 19
        elif dataset == "PROTEINS":
            border = 54
        elif dataset == "IMDBBINARY":
            border = 25
        elif dataset == "DD":
            border = 395
        elif dataset == "FRANK":
            border = 22
        elif dataset == "REDDIT":
            border = 469
        elif dataset == "COLLAB":
            border = 91
        else:
            raise ValueError("Invalid dataset")

        for i in range(len(graphs)):
            if graphs[i].g.number_of_nodes() >= border:
                graphs[i].nodegroup += 1

        subgraph_sample(dataset, graphs, border)
