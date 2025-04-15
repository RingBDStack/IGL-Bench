from .graph_clf import GraphClf
import torch.nn.functional as F
import torch
from sklearn.metrics import f1_score, accuracy_score,balanced_accuracy_score,roc_auc_score
import numpy as np
import os

class Model(object):
    def __init__(self, config):
        self.config = config
        self.criterion = F.nll_loss
        
        self.score_func = accuracy
        self.wf1 = wf1
        self.mf1 = mf1
        self.bacc = bacc
        self.auroc = auroc
        self.metric_name = 'acc'
        
        self._init_new_network()
        self._init_optimizer()
        
    def _init_new_network(self):
        self.network = GraphClf(self.config)
        
    def _init_optimizer(self):
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        
    def save(self, dirname):
        params = {
            'state_dict': {
                'network': self.network.state_dict(),
            },
            'config': self.config,
            'dir': dirname,
        }
        try:
            torch.save(params, os.path.join(dirname, "params.saved"))
        except BaseException:
            print('[ WARN: Saving failed... continuing anyway. ]')
            
    def init_saved_network(self, saved_dir):
        fname = os.path.join(saved_dir, "params.saved")
        print('[ Loading saved models %s ]' % fname)
        saved_params = torch.load(fname, map_location=lambda storage, loc: storage)
        self.state_dict = saved_params['state_dict']

        self.network = GraphClf(self.config)

        if self.state_dict:
            merged_state_dict = self.network.state_dict()
            for k, v in self.state_dict['network'].items():
                if k in merged_state_dict:
                    merged_state_dict[k] = v
            self.network.load_state_dict(merged_state_dict)
            
    def reset_parameters(self):
        print("[ Resetting model parameters ]")
        # Reinitialize the network
        self._init_new_network()

        # Reinitialize optimizer and scheduler
        self._init_optimizer()
        
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