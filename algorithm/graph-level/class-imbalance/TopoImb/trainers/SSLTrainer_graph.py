import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

from tqdm import tqdm
import numpy as np
import ipdb

# import datasets.data_utils as data_util
import utils
import models.models
from models.layers import MLP

import trainers.trainer as trainer

class TopoGraphTrainer(trainer.Trainer):
    '''for node classification
    
    '''
    def __init__(self, args, model, weight=1.0, dataset=None, out_size=None,shared_encoder=True):
        super().__init__(args, model, weight)        

        self.args = args
        self.shared_encoder = shared_encoder
        if shared_encoder:#provided model is the shared feature encoder, and trainer-specific classifier is required for each task
            self.in_dim = trainer.cal_feat_dim(args)
            if out_size is None:
                out_size = dataset[0].y.max().item()+1
            self.classifier = MLP(in_feat=self.in_dim, hidden_size=args.nhid, out_size=out_size, layers=args.nlayer)
            if args.cuda:
                self.classifier.cuda()
            self.classifier_opt = optim.Adam(self.classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
            self.models.append(self.classifier)
            self.models_opt.append(self.classifier_opt)


    def train_step(self, data, epoch, label=None):
        for i, model in enumerate(self.models):
            model.train()
            self.models_opt[i].zero_grad()

        if label is None:
            ipdb.set_trace()
            label = data.topo_label

        if self.shared_encoder:
            output = self.get_em(data)
            output = self.models[-1](output)
        else:
            output = self.models[0](data.x, data.edge_index, batch=data.batch)

        #output = F.log_softmax(output, dim=1)
        loss_train = F.nll_loss(output, label)
        acc_train = utils.accuracy(output, label)

        loss_all = loss_train*self.loss_weight
        for model in self.models:
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.0)

        loss_all.backward()
        
        for opt in self.models_opt:
            opt.step()
        
        # print('Epoch: {:05d}'.format(epoch+1),
        #       'loss_train: {:.4f}'.format(loss_train.item()),
        #     'acc_train: {:.4f}'.format(acc_train.item()))
        
        log_info = {'loss_train': loss_train.item(), 'acc_train': acc_train.item(), }

        return log_info

    def test(self, data, label=None):
        for i, model in enumerate(self.models):
            model.eval()

        if label is None:
            ipdb.set_trace()
            label = data.topo_label

        if self.shared_encoder:
            output = self.get_em(data)
            output = self.models[-1](output)
        else:
            output = self.models[0](data.x, data.edge_index, batch=data.batch)

        loss_test = F.nll_loss(output, label)
        acc_test = utils.accuracy(output, label)

        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))

        #utils.print_class_acc(output, label, pre='test')
        
        roc_test, macroF_test = utils.Roc_F(output, label, pre='test')
        
        log_info = {'loss_test': loss_test.item(), 'acc_test': acc_test.item(), 'roc_test': roc_test, 'macroF_test': macroF_test}

        return log_info

    def get_em(self, data):
        output = self.models[0].embedding(data.x, data.edge_index, batch=data.batch, return_list=False, graph_level=True)

        return output

