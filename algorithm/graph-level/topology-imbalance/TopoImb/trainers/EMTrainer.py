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


class EMClsTrainer(trainer.Trainer):
    '''for node classification
    
    '''
    def __init__(self, args, model, weight=1.0, dataset=None):
        super().__init__(args, model, weight)        

        self.args = args

        re_param_list = []
        re_mem_list = []
        for name, param in model.named_parameters():
            if 'memory' in name:
                re_mem_list.append(param)
            else:
                re_param_list.append(param)
        self.model_opt = optim.Adam(re_param_list, lr=args.lr, weight_decay=args.weight_decay)
        self.model_mem_opt = optim.Adam(re_mem_list, lr=args.lr, weight_decay=args.weight_decay)
        
        train_mask, val_mask, test_mask = data_util.split_graph(dataset[0], train_ratio=args.sup_ratio,val_ratio=args.val_ratio, test_ratio=args.test_ratio)
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask


    def train_step(self, data, epoch):
        for i, model in enumerate(self.models):
            model.train()
            
        self.model_opt.zero_grad()

        output = self.models[0](data.x, data.edge_index)

        loss_train = F.nll_loss(output[self.train_mask], data.y[self.train_mask])
        acc_train = utils.accuracy(output[self.train_mask], data.y[self.train_mask])

        loss_all = loss_train*self.loss_weight
        loss_all.backward()
        for model in self.models:
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.0)

        self.model_opt.step()
        
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

    def train_mem_step(self, data, epoch):
        for i, model in enumerate(self.models):
            model.train()
            
        self.model_mem_opt.zero_grad()

        output = self.models[0](data.x, data.edge_index)

        loss_train = F.nll_loss(output[self.train_mask], data.y[self.train_mask])
        acc_train = utils.accuracy(output[self.train_mask], data.y[self.train_mask])

        loss_all = loss_train*self.loss_weight
        loss_all.backward()
        for model in self.models:
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.0)

        self.model_mem_opt.step()
        
        #ipdb.set_trace()

        loss_val = F.nll_loss(output[self.val_mask], data.y[self.val_mask])
        acc_val = utils.accuracy(output[self.val_mask], data.y[self.val_mask])
        #utils.print_class_acc(output[self.val_mask], data.y[self.val_mask])
        roc_val, macroF_val = utils.Roc_F(output[self.val_mask], data.y[self.val_mask])

        
        print('Epoch: {:05d}'.format(epoch+1),
              'loss_train_mem: {:.4f}'.format(loss_train.item()),
            'acc_train_mem: {:.4f}'.format(acc_train.item()),
            'loss_val_mem: {:.4f}'.format(loss_val.item()),
            'acc_val_mem: {:.4f}'.format(acc_val.item()))
        
        log_info = {'loss_train_mem': loss_train.item(), 'acc_train_mem': acc_train.item(),
                     'loss_val_mem': loss_val.item(), 'acc_val_mem': acc_val.item(), 'roc_val_mem': roc_val, 'macroF_val_mem': macroF_val }
        
        return log_info

    def train_all_step(self, data, epoch):
        for i, model in enumerate(self.models):
            model.train()
            
        self.models_opt[0].zero_grad()

        output = self.models[0](data.x, data.edge_index)

        loss_train = F.nll_loss(output[self.train_mask], data.y[self.train_mask])
        acc_train = utils.accuracy(output[self.train_mask], data.y[self.train_mask])

        loss_all = loss_train*self.loss_weight
        loss_all.backward()
        for model in self.models:
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.0)

        self.models_opt[0].step()
        
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

    def test_all_topo(self, data, topo_label, only_test=False, return_weights=False, return_sims=False):
        #both data and topo_label should be in torch.tensor
        #only_test = True: examine only test data; only_test = False: examine all data

        for i, model in enumerate(self.models):
            model.eval()

        output = self.models[0](data.x, data.edge_index)
        _, sim_list = self.models[0].predict_weight(data.x, data.edge_index, return_sim=True)

        topo_sim_dict_list = []#show topo-group-wise memory cell selection behavior
        if only_test:
            topo_ac_dict = utils.grouped_accuracy(output[self.test_mask].cpu().detach().numpy(), data.y[self.test_mask].cpu().numpy(), topo_label[self.test_mask].cpu().numpy())
            for sim in sim_list:
                topo_sim_dict = utils.grouped_measure(sim[self.test_mask].cpu().detach().numpy(), topo_label[self.test_mask].cpu().numpy())
                topo_sim_dict_list.append(topo_sim_dict)
        else:
            topo_ac_dict = utils.grouped_accuracy(output.cpu().detach().numpy(), data.y.cpu().numpy(), topo_label.cpu().numpy())
            for sim in sim_list:
                topo_sim_dict = utils.grouped_measure(sim.cpu().detach().numpy(), topo_label.cpu().numpy())
                topo_sim_dict_list.append(topo_sim_dict)

        if return_sims:
            if return_weights:
                return topo_ac_dict, _, topo_sim_dict_list
            else:
                return topo_ac_dict, topo_sim_dict_list
        else:
            if return_weights:
                return topo_ac_dict, _
            else:
                return topo_ac_dict

    def get_emb(self, data):
        for i, model in enumerate(self.models):
            model.eval()

        with torch.no_grad():
            emb_list = self.models[0].embedding(data.x, data.edge_index, return_list=True)

        return emb_list


class EMGClsTrainer(trainer.Trainer):
    '''for graph classification
    
    '''
    def __init__(self, args, model, weight=1.0, dataset=None):
        super().__init__(args, model, weight)        

        self.args = args
        
        re_param_list = []
        re_mem_list = []
        for name, param in model.named_parameters():
            if 'memory' in name:
                re_mem_list.append(param)
            else:
                re_param_list.append(param)
        self.model_opt = optim.Adam(re_param_list, lr=args.lr, weight_decay=args.weight_decay)
        self.model_mem_opt = optim.Adam(re_mem_list, lr=args.lr, weight_decay=args.weight_decay)

    def train_step(self, data, epoch):
        for i, model in enumerate(self.models):
            model.train()
            
        self.model_opt.zero_grad()

        
        output = self.models[0](data.x, data.edge_index, batch=data.batch)

        loss_train = F.nll_loss(output, data.y)
        acc_train = utils.accuracy(output, data.y)

        loss_all = loss_train*self.loss_weight
        loss_all.backward()
        for model in self.models:
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.0)

        self.model_opt.step()
        
        #ipdb.set_trace()

        #utils.print_class_acc(output, data.y)
        #roc_train, macroF_train = utils.Roc_F(output, data.y)

        '''
        print('Epoch: {:05d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
            'acc_train: {:.4f}'.format(acc_train.item()),)
        '''
        log_info = {'loss_train': loss_train.item(), 'acc_train': acc_train.item(),
                    }

        return log_info

    def train_mem_step(self, data, epoch):
        for i, model in enumerate(self.models):
            model.train()

        self.model_mem_opt.zero_grad()

        output = self.models[0](data.x, data.edge_index, batch=data.batch)

        loss_train = F.nll_loss(output, data.y)
        acc_train = utils.accuracy(output, data.y)

        loss_all = loss_train*self.loss_weight
        loss_all.backward()
        for model in self.models:
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.0)

        self.model_mem_opt.step()
        
        #ipdb.set_trace()

        #utils.print_class_acc(output, data.y)
        #roc_train, macroF_train = utils.Roc_F(output, data.y)

        '''
        print('Epoch: {:05d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
            'acc_train: {:.4f}'.format(acc_train.item()),)
        '''
        log_info = {'loss_train_mem': loss_train.item(), 'acc_train_mem': acc_train.item(),
                    }

        return log_info

    def train_all_step(self, data, epoch):
        for i, model in enumerate(self.models):
            model.train()

        self.models_opt[0].zero_grad()

        output = self.models[0](data.x, data.edge_index, batch=data.batch)

        loss_train = F.nll_loss(output, data.y)
        acc_train = utils.accuracy(output, data.y)

        loss_all = loss_train*self.loss_weight
        loss_all.backward()
        for model in self.models:
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.0)

        self.models_opt[0].step()
        
        #ipdb.set_trace()

        #utils.print_class_acc(output, data.y)
        #roc_train, macroF_train = utils.Roc_F(output, data.y)

        '''
        print('Epoch: {:05d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
            'acc_train: {:.4f}'.format(acc_train.item()),)
        '''
        log_info = {'loss_train_mem': loss_train.item(), 'acc_train_mem': acc_train.item(),
                    }

        return log_info

    def test(self, data):
        for i, model in enumerate(self.models):
            model.eval()

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

    def test_all_topo(self, data, topo_label, only_test=False, return_weights=False, return_sims=False):
        #both data and topo_label should be in torch.tensor
        #only_test = True: examine only test data; only_test = False: examine all data
        for i, model in enumerate(self.models):
            model.eval()

        output = self.models[0](data.x, data.edge_index, batch=data.batch)

        _, sim_list = self.models[0].predict_weight(data.x, data.edge_index, batch=data.batch, return_sim=True)
        

        topo_sim_dict_list = []#show topo-group-wise memory cell selection behavior
        if only_test:
            topo_ac_dict = utils.grouped_accuracy(output.cpu().detach().numpy(), data.y.cpu().numpy(), topo_label.cpu().numpy())
            for sim in sim_list:
                topo_sim_dict = utils.grouped_measure(sim.cpu().detach().numpy(), topo_label.cpu().numpy())
                topo_sim_dict_list.append(topo_sim_dict)
        else:
            topo_ac_dict = utils.grouped_accuracy(output.cpu().detach().numpy(), data.y.cpu().numpy(), topo_label.cpu().numpy())
            for sim in sim_list:
                topo_sim_dict = utils.grouped_measure(sim.cpu().detach().numpy(), topo_label.cpu().numpy())
                topo_sim_dict_list.append(topo_sim_dict)

        if return_sims:
            if return_weights:
                return topo_ac_dict, _, topo_sim_dict_list
            else:
                return topo_ac_dict, topo_sim_dict_list
        else:
            if return_weights:
                return topo_ac_dict, _
            else:
                return topo_ac_dict

    def get_emb(self, data):
        for i, model in enumerate(self.models):
            model.eval()

        with torch.no_grad():
            emb_list = self.models[0].embedding(data.x, data.edge_index, batch=data.batch, return_list=True, graph_level=True)

        return emb_list