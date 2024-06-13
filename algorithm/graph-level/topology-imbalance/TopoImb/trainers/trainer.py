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


def cal_feat_dim(args): 
    '''get dimension of obtained embedding feature
    Args: 
        args:
    '''

    emb_dim = args.nhid
    if args.datatype == 'graph':
        emb_dim = emb_dim * 2
    if True: #res is defaulted to True
        emb_dim = emb_dim*args.nlayer

    return emb_dim


class Trainer(object):
    def __init__(self, args, model, weight):#
        self.args = args

        self.loss_weight = weight
        self.models = []
        self.models.append(model)

        self.models_opt = []
        for model in self.models:
            self.models_opt.append(optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay))

    def train_step(self, data):#pre_adj corresponds to adj used for generating ssl signal
        raise NotImplementedError('train not implemented for base class')

    def inference(self, data):
        raise NotImplementedError('infer not implemented for base class')

    def get_em(self, data):
        output = self.models[0].embedding(data.x, data.edge_index)

        return output

        
class ClsTrainer(Trainer):
    '''for node classification
    
    '''
    def __init__(self, args, model, weight=1.0, dataset=None):
        super().__init__(args, model, weight)        

        self.args = args
        
        if args.shared_encoder:#provided model is the shared feature encoder, and trainer-specific classifier is required for each task
            self.in_dim = cal_feat_dim(args)
            self.classifier = models.MLP(in_feat=self.in_dim, hidden_size=args.nhid, out_size=dataset[0].y.max().item() + 1, layers=args.cls_layer)
            if args.cuda:
                self.classifier.cuda()
            self.classifier_opt = optim.Adam(self.classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
            self.models.append(self.classifier)
            self.models_opt.append(self.classifier_opt)

        
        train_mask, val_mask, test_mask = data_util.split_graph(dataset[0], train_ratio=args.sup_ratio,val_ratio=args.val_ratio, test_ratio=args.test_ratio)
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask


    def train_step(self, data, epoch):
        for i, model in enumerate(self.models):
            model.train()
            self.models_opt[i].zero_grad()

        if self.args.shared_encoder:
            output = self.get_em(data)
            output = self.models[-1](output)
        else:
            output = self.models[0](data.x, data.edge_index)

        loss_train = F.nll_loss(output[self.train_mask], data.y[self.train_mask])
        acc_train = utils.accuracy(output[self.train_mask], data.y[self.train_mask])

        loss_all = loss_train*self.loss_weight
        loss_all.backward()
        for model in self.models:
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.0)

        for opt in self.models_opt:
            opt.step()
        
        #ipdb.set_trace()

        loss_val = F.nll_loss(output[self.val_mask], data.y[self.val_mask])
        acc_val = utils.accuracy(output[self.val_mask], data.y[self.val_mask])
        #utils.print_class_acc(output[self.val_mask], data.y[self.val_mask])
        roc_val, macroF_val = utils.Roc_F(output[self.val_mask], data.y[self.val_mask])

        
        print('Epoch: {:05d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
            'acc_train: {:.4f}'.format(acc_train.item()),
            'loss_val: {:.4f}'.format(loss_val.item()),
            'acc_val: {:.4f}'.format(acc_val.item()))
        
        log_info = {'loss_train': loss_train.item(), 'acc_train': acc_train.item(),
                     'loss_val': loss_val.item(), 'acc_val': acc_val.item(), 'roc_val': roc_val, 'macroF_val': macroF_val }

        return log_info

    def test(self, data):
        for i, model in enumerate(self.models):
            model.eval()


        if self.args.shared_encoder:
            output = self.get_em(data)
            output = self.models[-1](output)
        else:
            output = self.models[0](data.x, data.edge_index)

        loss_test = F.nll_loss(output[self.test_mask], data.y[self.test_mask])
        acc_test = utils.accuracy(output[self.test_mask], data.y[self.test_mask])

        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))

        utils.print_class_acc(output[self.test_mask], data.y[self.test_mask], pre='test')
        
        roc_test, macroF_test = utils.Roc_F(output[self.test_mask], data.y[self.test_mask], pre='test')
        
        log_info = {'loss_test': loss_test.item(), 'acc_test': acc_test.item(), 'roc_test': roc_test, 'macroF_test': macroF_test}

        return log_info

    def test_all_topo(self, data, topo_label, only_test=False):
        #both data and topo_label should be in torch.tensor
        #only_test = True: examine only test data; only_test = False: examine all data
        for i, model in enumerate(self.models):
            model.eval()

        if self.args.shared_encoder:
            output = self.get_em(data)
            output = self.models[-1](output)
        else:
            output = self.models[0](data.x, data.edge_index)

        if only_test:
            topo_ac_dict = utils.grouped_accuracy(output[self.test_mask].cpu().detach().numpy(), data.y[self.test_mask].cpu().numpy(), topo_label[self.test_mask].cpu().numpy())
        else:
            topo_ac_dict = utils.grouped_accuracy(output.cpu().detach().numpy(), data.y.cpu().numpy(), topo_label.cpu().numpy())

        return topo_ac_dict

class GClsTrainer(Trainer):
    '''for graph classification
    
    '''
    def __init__(self, args, model, weight=1.0, dataset=None):
        super().__init__(args, model, weight)        

        self.args = args
        
        if args.shared_encoder:#provided model is the shared feature encoder, and trainer-specific classifier is required for each task
            self.in_dim = cal_feat_dim(args)
            self.classifier = models.MLP(in_feat=self.in_dim, hidden_size=args.nhid, out_size=dataset.y.max().item() + 1, layers=args.cls_layer)
            if args.cuda:
                self.classifier.cuda()
            self.classifier_opt = optim.Adam(self.classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
            self.models.append(self.classifier)
            self.models_opt.append(self.classifier_opt)


    def train_step(self, data, epoch):
        for i, model in enumerate(self.models):
            model.train()
            self.models_opt[i].zero_grad()

        if self.args.shared_encoder:#not tested yet
            ipdb.set_trace()
            output = self.get_em(data)#need to include batch informaiton
            output = self.models[-1](output)
        else:
            output = self.models[0](data.x, data.edge_index, batch=data.batch)

        loss_train = F.nll_loss(output, data.y)
        acc_train = utils.accuracy(output, data.y)

        loss_all = loss_train*self.loss_weight
        loss_all.backward()
        for model in self.models:
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.0)

        for opt in self.models_opt:
            opt.step()
        
        #ipdb.set_trace()

        #utils.print_class_acc(output, data.y)
        #roc_train, macroF_train = utils.Roc_F(output, data.y)


        print('Epoch: {:05d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
            'acc_train: {:.4f}'.format(acc_train.item()),)

        log_info = {'loss_train': loss_train.item(), 'acc_train': acc_train.item(),
                    }

        return log_info

    def test(self, data):
        for i, model in enumerate(self.models):
            model.eval()

        if self.args.shared_encoder: #not tested yet
            ipdb.set_trace()
            output = self.get_em(data)#need to revise for graphs
            output = self.models[-1](output)
        else:
            output = self.models[0](data.x, data.edge_index, batch=data.batch)

        loss_test = F.nll_loss(output, data.y)
        acc_test = utils.accuracy(output, data.y)

        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))

        utils.print_class_acc(output, data.y, pre='test')
        
        roc_test, macroF_test = utils.Roc_F(output, data.y, pre='test')
        
        log_info = {'loss_test': loss_test.item(), 'acc_test': acc_test.item(), 'roc_test': roc_test, 'macroF_test': macroF_test}

        return log_info

    def test_all_topo(self, data, topo_label, only_test=False):
        #both data and topo_label should be in torch.tensor
        #only_test = True: examine only test data; only_test = False: examine all data
        for i, model in enumerate(self.models):
            model.eval()

        if self.args.shared_encoder:
            output = self.get_em(data)
            output = self.models[-1](output)
        else:
            output = self.models[0](data.x, data.edge_index)

        if only_test:
            topo_ac_dict = utils.grouped_accuracy(output.cpu().detach().numpy(), data.y.cpu().numpy(), topo_label.cpu().numpy())
        else:
            topo_ac_dict = utils.grouped_accuracy(output.cpu().detach().numpy(), data.y.cpu().numpy(), topo_label.cpu().numpy())

        return topo_ac_dict

    def get_em(self, data):
        output = self.models[0].embedding(data.x, data.edge_index, batch=data.batch, return_list=False, graph_level=True)

        return output
        


      
class MLPTrainer(Trainer):
    '''for node classification
    
    '''
    def __init__(self, args, model, weight=1.0, dataset=None, batch_size=32, train_ratio=0.7):
        super().__init__(args, model, weight)        

        self.args = args
        
        train_mask, val_mask, test_mask = data_util.split_graph(dataset[0], train_ratio=train_ratio,val_ratio=0.1, test_ratio=0.2)
        self.train_idx = train_mask.nonzero().squeeze()
        self.val_idx = val_mask.nonzero().squeeze()
        self.test_idx = test_mask.nonzero().squeeze()

        self.batch_size = batch_size

    def train_step(self, data, pre_adj):
        raise NotImplementedError('train one epoch for MLP model is not implemented yet')

    def train_batch(self, data, epoch, train_batch):
        for i, model in enumerate(self.models):
            model.train()
            self.models_opt[i].zero_grad()

        selected_idx = self.train_idx[train_batch*self.batch_size:(train_batch+1)*self.batch_size]

        output = self.models[0](data.x[selected_idx])

        loss_train = F.nll_loss(output, data.y[selected_idx])
        acc_train = utils.accuracy(output, data.y[selected_idx])

        loss_all = loss_train*self.loss_weight
        loss_all.backward()
        for model in self.models:
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.0)

        for opt in self.models_opt:
            opt.step()
        
        #ipdb.set_trace()

        output = self.models[0](data.x[self.val_idx])
        loss_val = F.nll_loss(output, data.y[self.val_idx])
        acc_val = utils.accuracy(output, data.y[self.val_idx])
        roc_val, macroF_val = utils.Roc_F(output, data.y[self.val_idx])

        
        print('Epoch: {:05d}'.format(epoch+1),
            'batch: {}'.format(train_batch),
            'loss_train: {:.4f}'.format(loss_train.item()),
            'acc_train: {:.4f}'.format(acc_train.item()),
            'loss_val: {:.4f}'.format(loss_val.item()),
            'acc_val: {:.4f}'.format(acc_val.item()))
        
        log_info = {'loss_train': loss_train.item(), 'acc_train': acc_train.item(),
                     'loss_val': loss_val.item(), 'acc_val': acc_val.item(), 'roc_val': roc_val, 'macroF_val': macroF_val }

        return log_info

    def test(self, data):
        for i, model in enumerate(self.models):
            model.eval()

        output = self.models[0](data.x[self.test_idx])

        loss_test = F.nll_loss(output, data.y[self.test_idx])
        acc_test = utils.accuracy(output, data.y[self.test_idx])

        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))

        utils.print_class_acc(output, data.y[self.test_idx], pre='test')
        
        roc_test, macroF_test = utils.Roc_F(output, data.y[self.test_idx], pre='test')
        
        log_info = {'loss_test': loss_test.item(), 'acc_test': acc_test.item(), 'roc_test': roc_test, 'macroF_test': macroF_test}

        return log_info