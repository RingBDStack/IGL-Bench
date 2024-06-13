import os
import torch
import numpy as np
import networkx as nx
# import torchvision

class console_log(object):
    def __init__(self, logs_path='./'):
        self.logs_path = logs_path
        if not os.path.exists(self.logs_path):
            os.makedirs(self.logs_path, exist_ok=True)

    def write_log(self, log_str=None, should_print=True, prefix='console', end='\n'):
        with open(os.path.join(self.logs_path, '%s.log' % prefix), 'a') as fout:
            fout.write(log_str + end)
            fout.flush()
        if should_print:
            print(log_str)


def get_lord_error_fn(logits, Y, ord):
    errors = torch.nn.functional.softmax(logits, dim=1) - Y
    scores = np.linalg.norm(errors.detach().cpu().numpy(), ord=ord, axis=-1)
    return scores 

def get_grad_norm_fn(loss_grads):
    scores = np.linalg.norm(loss_grads, axis=-1)
    return scores 

def get_subset(rate, ranked_list):
   sample_size = int(rate * len(ranked_list))
   return ranked_list[:sample_size]


class imbalance_node_dataset(object):
    def __init__(self, data, imb_type='exp', imb_factor=0.01, cls_num=10):
        self.cls_num = cls_num
        self.data = data
        node_num_list = self.get_node_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(node_num_list)

    def get_node_num_per_cls(self, cls_num, imb_type, imb_factor, train_mask):
        node_max = len(self.data.x[train_mask]) / cls_num
        node_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = node_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                node_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                node_num_per_cls.append(int(node_max))
            for cls_idx in range(cls_num // 2):
                node_num_per_cls.append(int(node_max * imb_factor))
        else:
            node_num_per_cls.extend([int(node_max)] * cls_num)
        return node_num_per_cls

    def gen_imbalanced_data(self, node_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.data.y, dtype=np.int64)
        classes = np.unique(targets_np)
         
        self.num_per_cls_dict = dict()
        for the_class, the_node_num in zip(classes, node_num_per_cls):
            self.num_per_cls_dict[the_class] = the_node_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_node_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_node_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

if __name__ == '__main__':
    G = nx.path_graph(4)   
    e = (1, 2)
    G.remove_edge(*e)   
    G = G
    e = (2, 3, {"weight": 7})   
    G.remove_edge(*e[:2])   
