import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score,balanced_accuracy_score,roc_auc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

from .models.graph_clf import GraphClf
from .utils import constants as Constants
from .utils.radam import RAdam


class Model(object):
    def __init__(self, config, train_set=None):
        self.config = config
        self.net_module = GraphClf
        self.criterion = F.nll_loss
        self.score_func = accuracy
        self.wf1 = wf1
        self.mf1 = mf1
        self.bacc = bacc
        self.auroc = auroc
        self.metric_name = 'acc'

        self._init_new_network()

        num_params = 0
        print("[ Model Parameters ]")
        for name, p in self.network.named_parameters():
            print('{}: {}'.format(name, str(p.size())))
            num_params += p.numel()

        print('# Parameters = {}\n'.format(num_params))
        print("[ End Model Parameters ]")
        self._init_optimizer()


    def init_saved_network(self, saved_dir):
        fname = os.path.join(saved_dir, Constants._SAVED_WEIGHTS_FILE)
        print('[ Loading saved models %s ]' % fname)
        saved_params = torch.load(fname, map_location=lambda storage, loc: storage)
        self.state_dict = saved_params['state_dict']

        self.network = self.net_module(self.config)

        if self.state_dict:
            merged_state_dict = self.network.state_dict()
            for k, v in self.state_dict['network'].items():
                if k in merged_state_dict:
                    merged_state_dict[k] = v
            self.network.load_state_dict(merged_state_dict)


    def _init_new_network(self):
        self.network = self.net_module(self.config)


    def _init_optimizer(self):
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if self.config['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(parameters, self.config['learning_rate'],
                                       momentum=self.config['momentum'],
                                       weight_decay=self.config['weight_decay'])
        elif self.config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(parameters, lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])
        elif self.config['optimizer'] == 'adamax':
            self.optimizer = optim.Adamax(parameters, lr=self.config['learning_rate'])
        elif self.config['optimizer'] == 'radam':
            self.optimizer = RAdam(parameters, lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])
        else:
            raise RuntimeError('Unsupported optimizer: %s' % self.config['optimizer'])
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=self.config['lr_reduce_factor'], \
                    patience=self.config['lr_patience'], verbose=True)


    def save(self, dirname):
        params = {
            'state_dict': {
                'network': self.network.state_dict(),
            },
            'config': self.config,
            'dir': dirname,
        }
        try:
            torch.save(params, os.path.join(dirname, Constants._SAVED_WEIGHTS_FILE))
        except BaseException:
            print('[ WARN: Saving failed... continuing anyway. ]')


    def clip_grad(self):
        if self.config['grad_clipping']:
            parameters = [p for p in self.network.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(parameters, self.config['grad_clipping'])


def accuracy(labels, output):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum().item()
    return correct / len(labels)


def wf1(labels, output):
    pred = output.cpu().max(1)[1].numpy()
    labels = labels.cpu().numpy()
    return f1_score(labels, pred, average='weighted')


def mf1(labels, output):
    pred = output.cpu().max(1)[1].numpy()
    labels = labels.cpu().numpy()
    return f1_score(labels, pred, average='macro')

def bacc(labels, output):
    pred = output.cpu().max(1)[1].numpy()
    labels = labels.cpu().numpy()
    return balanced_accuracy_score(labels, pred)

def auroc(labels, output):
    labels = labels.cpu().numpy()  
    output = output.cpu().detach().numpy()  

    n_classes = output.shape[1]  
    labels_binary = np.eye(n_classes)[labels]

    auroc = roc_auc_score(labels_binary, output, multi_class='ovr', average='macro')
    return auroc