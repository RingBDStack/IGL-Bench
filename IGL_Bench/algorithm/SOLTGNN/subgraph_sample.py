import numpy as np
import os


subgraph_default_border = {
    'PTC_MR': 19,
    "PROTEINS": 54,
    "IMDB-BINARY": 25,
    "DD": 395,
    "FRANKENSTEIN": 22,
    "REDDIT": 469,
    "COLLAB": 91
}


def subgraph_sample(dataset, graph_list, nums=500):
    np.random.seed(0)
    border = subgraph_default_border.get(dataset, 0)
    for i in range(len(graph_list)):
        if graph_list[i].g.number_of_nodes() >= border:
            graph_list[i].nodegroup += 1
    sample_path = os.path.join(os.path.dirname(__file__), f'sampling/{dataset}/sampling.txt')
    with open(sample_path, 'w') as f:
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


if __name__ == '__main__':
    pass
