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


class ReweighterTrainer(trainer.Trainer):
    def __init__(self, args, model, reweighter, weight=1.0, dataset=None, weights = None):
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

        self.reweighter = reweighter
        if reweighter is not None:
            self.reweight_lr = args.reweight_lr
            re_param_list = []
            re_mem_list = []
            for name, param in self.reweighter.named_parameters():
                if 'memory' in name:
                    re_mem_list.append(param)
                else:
                    re_param_list.append(param)
            self.reweighter_opt = optim.Adam(re_param_list, lr=args.reweight_lr, weight_decay=args.weight_decay)
            if len(re_mem_list) !=0:
                self.reweighter_mem_opt = optim.Adam(re_mem_list, lr=args.reweight_lr, weight_decay=args.weight_decay)
        else:
            assert weights is not None, "weights should not be None if no reweighter is used in ReweighterTrainer"
            self.reweights = dataset[0].x.new(weights)

        self.adv_step = args.adv_step

        if args.split==0 or args.dataset=='ImbNode':
            #train_mask, val_mask, test_mask = data_util.split_graph(dataset[0], train_ratio=args.sup_ratio,val_ratio=args.val_ratio, test_ratio=args.test_ratio, imb_ratio=args.intra_im_ratio)
            train_mask, val_mask, test_mask = data_util.split_graph(dataset[0], train_ratio=args.sup_ratio,val_ratio=args.val_ratio, test_ratio=args.test_ratio)
        else:
            train_mask, val_mask, test_mask = data_util.split_graph_arti(dataset[0], train_ratio=args.sup_ratio,val_ratio=args.val_ratio, test_ratio=args.test_ratio, imb_ratio=args.intra_im_ratio)
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask

        self.adv_step = args.adv_step

    def train_step(self, data, epoch):
        for i, model in enumerate(self.models):
            model.train()
            self.models_opt[i].zero_grad()
        if self.reweighter is not None:
            self.reweighter.train()

        if self.args.shared_encoder:
            output = self.get_em(data)
            output = self.models[-1](output)
        else:
            output = self.models[0](data.x, data.edge_index)

        #-----------------
        #update predictor
        #-----------------
        if self.reweighter is not None:
            with torch.no_grad():
                all_weights = self.reweighter.predict_weight(data.x, data.edge_index, return_sim=False).squeeze()
                weights = all_weights[self.train_mask]
                #weight_avg = weights.detach().mean()
                #weights = weights/(weight_avg+0.0001)

            loss_train_ori = F.nll_loss(output[self.train_mask], data.y[self.train_mask])
            acc_train = utils.accuracy(output[self.train_mask], data.y[self.train_mask])

            loss_train_ins = F.nll_loss(output[self.train_mask], data.y[self.train_mask], reduction='none')
            loss_train_re = torch.mean(weights.detach()*loss_train_ins)
        else:
            loss_train_ori = F.nll_loss(output[self.train_mask], data.y[self.train_mask])
            acc_train = utils.accuracy(output[self.train_mask], data.y[self.train_mask])
            loss_train_re = F.nll_loss(output[self.train_mask], data.y[self.train_mask], weight=self.reweights.to(data.y.device))

        assert 0<=self.loss_weight<=1, "reweight_weight should within [0,1]"
        loss_all = loss_train_re*self.loss_weight + loss_train_ori*(1-self.loss_weight)

        loss_all.backward()
        for model in self.models:
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.0)

        for opt in self.models_opt:
            opt.step()
        
        #-----------------
        #update reweighter
        #-----------------
        if self.reweighter is not None:
            for step in range(self.adv_step):
                self.reweighter_opt.zero_grad()

                all_weights = self.reweighter.predict_weight(data.x, data.edge_index, return_sim=False).squeeze()
                weights = all_weights[self.train_mask]
                with torch.no_grad():
                    loss_weight_back = -F.nll_loss(output[self.train_mask].detach(), data.y[self.train_mask], reduction='none')
                loss_weight = torch.mean(weights*(loss_weight_back.detach()))
                weight_reg = loss_weight_back.detach().mean().abs()*weights.mean()

                loss_weight_all = loss_weight + weight_reg

                #re_output = self.reweighter(data.x, data.edge_index)
                #loss_reweight_cls = F.nll_loss(re_output[self.train_mask], data.y[self.train_mask])
                loss_weight_all = loss_weight_all

                loss_weight_all = loss_weight_all

                loss_weight_all.backward()
                
                torch.nn.utils.clip_grad_norm_(self.reweighter.parameters(),2.0)
                self.reweighter_opt.step()
        else:
            loss_weight = loss_train_ori
            weight_reg = loss_train_ori

        #-----------------
        #show performance
        #-----------------
        #ipdb.set_trace()
        loss_val = F.nll_loss(output[self.val_mask], data.y[self.val_mask])
        acc_val = utils.accuracy(output[self.val_mask], data.y[self.val_mask])
        #utils.print_class_acc(output[self.val_mask], data.y[self.val_mask])
        #roc_val, macroF_val = utils.Roc_F(output[self.val_mask], data.y[self.val_mask])
        
        print('Epoch: {:05d}'.format(epoch+1),
              'loss_train_ori: {:.4f}'.format(loss_train_ori.item()),
              'loss_train_re: {:.4f}'.format(loss_train_re.item()),
            'acc_train: {:.4f}'.format(acc_train.item()),
            'loss_val: {:.4f}'.format(loss_val.item()),
            'acc_val: {:.4f}'.format(acc_val.item()),
            'loss_weight: {:.4f}'.format(loss_weight.item()),
            'weight_reg: {:.4f}'.format(weight_reg.item()),
            )
        
        log_info = {'loss_train_ori': loss_train_ori.item(), 'loss_train_reweighted': loss_train_re.item(), 'acc_train': acc_train.item(),
                     'loss_val': loss_val.item(), 'acc_val': acc_val.item(), 
                     'loss_weight': loss_weight.item(), 'weight_reg': weight_reg.item(), }

        return log_info

    def train_mem_step(self, data, epoch):
        assert self.reweighter is not None, "train_mem_step() should not be called when reweighter is None"

        for i, model in enumerate(self.models):
            model.train()
            self.models_opt[i].zero_grad()
        self.reweighter.train()

        if self.args.shared_encoder:
            output = self.get_em(data)
            output = self.models[-1](output)
        else:
            output = self.models[0](data.x, data.edge_index)

        #-----------------
        #update reweighter
        #-----------------
        self.reweighter_mem_opt.zero_grad()

        all_weights = self.reweighter.predict_weight(data.x, data.edge_index, return_sim=False).squeeze()
        weights = all_weights[self.train_mask]
        with torch.no_grad():
            loss_weight_back = -F.nll_loss(output[self.train_mask].detach(), data.y[self.train_mask], reduction='none')
        loss_weight = torch.mean(weights*(loss_weight_back.detach()))
        weight_reg = loss_weight_back.detach().mean().abs()*weights.mean()
        loss_weight_all = loss_weight+weight_reg

        loss_weight_all.backward()
        
        torch.nn.utils.clip_grad_norm_(self.reweighter.parameters(),2.0)
        self.reweighter_mem_opt.step()

        #-----------------
        #show performance
        #-----------------
        #ipdb.set_trace()
        
        print('Epoch: {:05d}'.format(epoch+1),
            'loss_weight_mem: {:.4f}'.format(loss_weight.item()),
            'weight_reg_mem: {:.4f}'.format(weight_reg.item())
            )
        
        log_info = {'loss_weight_mem': loss_weight.item(), 'weight_reg_mem': weight_reg.item(), }

        return log_info

    def test(self, data, use_print=True):
        for i, model in enumerate(self.models):
            model.eval()

        if self.args.shared_encoder:
            output = self.get_em(data)
            output = self.models[-1](output)
        else:
            output = self.models[0](data.x, data.edge_index)

        #-----------------
        #test predictor
        #-----------------
        loss_test = F.nll_loss(output[self.test_mask], data.y[self.test_mask])
        acc_test = utils.accuracy(output[self.test_mask], data.y[self.test_mask])

        if use_print:
            print("Test set results:",
                "loss= {:.4f}".format(loss_test.item()),
                "accuracy= {:.4f}".format(acc_test.item()))
            utils.print_class_acc(output[self.test_mask], data.y[self.test_mask], pre='test')
        
        roc_test, macroF_test = utils.Roc_F(output[self.test_mask], data.y[self.test_mask], pre='test')
        
        if 'topo_label' in data:
            topo_macro_acc, topo_macro_auc, topo_macro_F = utils.groupewise_perform(output[self.test_mask], data.y[self.test_mask], data.topo_label[self.test_mask], pre='test')        
            log_info = {'loss_test': loss_test.item(), 'acc_test': acc_test.item(), 'roc_test': roc_test, 'macroF_test': macroF_test, 
                        'topo_macro_acc_test': topo_macro_acc, 'topo_macro_auc_test': topo_macro_auc, 'topo_macro_F_test': topo_macro_F}
        else:
            log_info = {'loss_test': loss_test.item(), 'acc_test': acc_test.item(), 'roc_test': roc_test, 'macroF_test': macroF_test}

        return log_info


    def test_all_topo(self, data, topo_label, only_test=False, only_train=False, return_weights=False, return_sims=False):
        #both data and topo_label should be in torch.tensor
        #only_test = True: examine only test data; only_test = False: examine all data
        #only_train = True: examine only train data

        for i, model in enumerate(self.models):
            model.eval()

        if self.args.shared_encoder:
            output = self.get_em(data)
            output = self.models[-1](output)
        else:
            output = self.models[0](data.x, data.edge_index)

        if self.reweighter is not None:
            if return_sims:
                all_weights, sim_list = self.reweighter.predict_weight(data.x, data.edge_index, return_sim=True)
            else:
                all_weights = self.reweighter.predict_weight(data.x, data.edge_index, return_sim=False)
            all_weights = all_weights.squeeze()
        else:
            all_weights = data.x.new(data.y.shape)
            for y in range(len(set(data.y.cpu().numpy()))):
                all_weights[data.y==y] = self.reweights[y]
            sim_list = [all_weights.new(data.y.shape).unsqueeze(-1).fill_(0)]

        all_weights = all_weights.squeeze()

        topo_sim_dict_list = []#show topo-group-wise memory cell selection behavior

        if only_test:
            sel_mask = self.test_mask
        elif only_train:
            sel_mask = self.train_mask
        else:
            sel_mask = self.test_mask.new(self.test_mask.shape).fill_(True)

        topo_ac_dict = utils.grouped_accuracy(output[sel_mask].cpu().detach().numpy(), data.y[sel_mask].cpu().numpy(), topo_label[sel_mask].cpu().numpy())
        topo_weight_dict = utils.grouped_measure(all_weights[sel_mask].cpu().detach().numpy(), topo_label[sel_mask].cpu().numpy())
        if return_sims:
            for sim in sim_list:
                topo_sim_dict = utils.grouped_measure(sim[sel_mask].cpu().detach().numpy(), topo_label[sel_mask].cpu().numpy())
                topo_sim_dict_list.append(topo_sim_dict)

        if return_sims:
            if return_weights:
                return topo_ac_dict, topo_weight_dict, topo_sim_dict_list
            else:
                return topo_ac_dict, topo_sim_dict_list
        else:
            if return_weights:
                return topo_ac_dict, topo_weight_dict
            else:
                return topo_ac_dict
