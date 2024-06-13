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

import trainers.trainer as trainer
from focal_loss import focal_loss

class BaselineGraphTrainer(trainer.Trainer):
    def __init__(self, args, model, method='no', minority_class=1, weight=1.0, dataset=None, weights=None):
        super().__init__(args, model, weight)        

        self.args = args
        
        if args.shared_encoder:#provided model is the shared feature encoder, and trainer-specific classifier is required for each task
            self.in_dim = trainer.cal_feat_dim(args)
            self.classifier = models.MLP(in_feat=self.in_dim, hidden_size=args.nhid, out_size=dataset[0].y.max().item() + 1, layers=args.cls_layer)
            if args.cuda:
                self.classifier.cuda()
            self.classifier_opt = optim.Adam(self.classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
            self.models.append(self.classifier)
            self.models_opt.append(self.classifier_opt)

        self.adv_step = args.adv_step
        self.method = method
        self.minority_class = minority_class
        

    def train_step(self, data, epoch):
        for i, model in enumerate(self.models):
            model.train()
            self.models_opt[i].zero_grad()

        labels = data.y
        if self.method == 'no':
            output = self.models[0](data.x, data.edge_index, batch=data.batch)
            loss_train_ori = F.nll_loss(output, data.y)
        elif self.method == 'reweight':
            weight = data.x.new((data.y.max().item()+1)).fill_(1)
            weight[-self.minority_class:] = 1/self.args.intra_im_ratio
            output = self.models[0](data.x, data.edge_index, batch=data.batch)
            loss_train_ori = F.nll_loss(output, data.y, weight=weight)
        elif self.method == 'focal':
            embed = self.models[0].embedding(data.x, data.edge_index, batch=data.batch,graph_level=True)
            pred = F.softmax(self.models[0].lin(embed), dim=-1)
            loss_train_ori = focal_loss(pred, data.y, alpha=0.5, gamma=2.0, reduction='mean')
            output = pred
        elif self.method == 'EMsmote':
            embed = self.models[0].embedding(data.x, data.edge_index, batch=data.batch, graph_level=True)
            embed, label_new, idx = utils.emb_smote(embed, data.y, portion=1, im_class_num=self.minority_class)
            pred = F.softmax(self.models[0].lin(embed), dim=-1)
            loss_train_ori = F.cross_entropy(pred, label_new)
            output = pred
            labels=label_new
        elif self.method == 'oversample':
            ipdb.set_trace()

        #-----------------
        #update predictor
        #-----------------
        acc_train = utils.accuracy(output, labels)


        loss_all = loss_train_ori

        loss_all.backward()
        for model in self.models:
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.0)

        for opt in self.models_opt:
            opt.step()
        

        #-----------------
        #show performance
        #-----------------
        #ipdb.set_trace()
        #utils.print_class_acc(output, data.y)
        #roc_train, macroF_train = utils.Roc_F(output, data.y)
        
        print('Epoch: {:05d}'.format(epoch+1),
              'loss_train ori: {:.4f}'.format(loss_train_ori.item()),
            'acc_train: {:.4f}'.format(acc_train.item()),
            )
        
        log_info = {'loss_train_ori': loss_train_ori.item(),'acc_train': acc_train.item() }

        return log_info

    def test(self, data):
        for i, model in enumerate(self.models):
            model.eval()

        if self.args.shared_encoder:
            output = self.get_em(data)
            output = self.models[-1](output)
        else:
            output = self.models[0](data.x, data.edge_index, batch=data.batch)

        #-----------------
        #test predictor
        #-----------------
        loss_test = F.nll_loss(output, data.y)
        acc_test = utils.accuracy(output, data.y)

        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))

        utils.print_class_acc(output, data.y, pre='test')
        
        roc_test, macroF_test = utils.Roc_F(output, data.y, pre='test')
        if 'topo_label' in data:
            topo_macro_acc, topo_macro_auc, topo_macro_F = utils.groupewise_perform(output, data.y, data.topo_label, pre='test')        
            log_info = {'loss_test': loss_test.item(), 'acc_test': acc_test.item(), 'roc_test': roc_test, 'macroF_test': macroF_test, 
                        'topo_macro_acc_test': topo_macro_acc, 'topo_macro_auc_test': topo_macro_auc, 'topo_macro_F_test': topo_macro_F}
        else:
            log_info = {'loss_test': loss_test.item(), 'acc_test': acc_test.item(), 'roc_test': roc_test, 'macroF_test': macroF_test}

        return log_info

    def test_all_topo(self, data, topo_label, only_test=False, only_train=False, return_weights=False, return_sims=False):
        #both data and topo_label should be in torch.tensor
        #only_test = True: examine only test data; only_test = False: examine all data
        for i, model in enumerate(self.models):
            model.eval()

        if self.args.shared_encoder:
            ipdb.set_trace()#not tested for batch yet
            output = self.get_em(data)
            output = self.models[-1](output)
        else:
            output = self.models[0](data.x, data.edge_index, batch=data.batch)


        
        topo_ac_dict = utils.grouped_accuracy(output.cpu().detach().numpy(), data.y.cpu().numpy(), topo_label.cpu().numpy())

        return topo_ac_dict
