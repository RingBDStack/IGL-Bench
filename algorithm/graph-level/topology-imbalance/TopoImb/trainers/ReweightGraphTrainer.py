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
from sklearn.metrics import balanced_accuracy_score, silhouette_score, f1_score, roc_auc_score, accuracy_score


class ReweighterGraphTrainer(trainer.Trainer):
    def __init__(self, args, model, reweighter, weight=1.0, dataset=None, weights=None):
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
        self.reweight_lr = args.reweight_lr

        re_param_list = []
        re_mem_list = []
        for name, param in self.reweighter.named_parameters():
            if 'memory' in name:
                re_mem_list.append(param)
            else:
                re_param_list.append(param)
        self.reweighter_opt = optim.Adam(re_param_list, lr=args.reweight_lr, weight_decay=args.weight_decay)
        self.reweighter_mem_opt = optim.Adam(re_mem_list, lr=args.reweight_lr, weight_decay=args.weight_decay)

        self.adv_step = args.adv_step
        

    def train_step(self, data, epoch):
        for i, model in enumerate(self.models):
            model.train()
            self.models_opt[i].zero_grad()
        self.reweighter.train()

        if self.args.shared_encoder:
            ipdb.set_trace()#not tested yet, for supporting batch information
            output = self.get_em(data)
            output = self.models[-1](output)
        else:
            output = self.models[0](data.x, data.edge_index, batch=data.batch)

        #-----------------
        #update predictor
        #-----------------
        with torch.no_grad():
            all_weights = self.reweighter.predict_weight(data.x, data.edge_index, batch=data.batch, return_sim=False).squeeze()
            weights = all_weights
            #weight_avg = weights.detach().mean()
            #weights = weights/(weight_avg+0.0001)

        loss_train_ori = F.nll_loss(output, data.y)
        acc_train = utils.accuracy(output, data.y)

        loss_train_ins = F.nll_loss(output, data.y, reduction='none')
        loss_train_re = torch.mean(weights.detach()*loss_train_ins)

        assert 0<=self.loss_weight<=1, "reweight_weight should within [0,1]"
        loss_all = loss_train_re*self.loss_weight + loss_train_ori*(1-self.loss_weight)

        for model in self.models:
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.0)
            
        loss_all.backward()

        for opt in self.models_opt:
            opt.step()
        
        #-----------------
        #update reweighter
        #-----------------
        for step in range(self.adv_step):
            self.reweighter_opt.zero_grad()
            all_weights = self.reweighter.predict_weight(data.x, data.edge_index,batch=data.batch, return_sim=False).squeeze()
            weights = all_weights
            with torch.no_grad():
                loss_weight_back = -F.nll_loss(output.detach(), data.y, reduction='none')
            loss_weight = torch.mean(weights*(loss_weight_back.detach()))
            weight_reg = loss_weight_back.detach().mean().abs()*weights.mean()
            loss_weight_all = loss_weight+weight_reg

            loss_weight_all = loss_weight


            re_output = self.reweighter(data.x, data.edge_index, batch=data.batch)
            loss_reweight_cls = F.nll_loss(re_output, data.y)
            loss_weight_all = loss_weight_all+loss_reweight_cls

            loss_weight_all = loss_weight_all

            loss_weight_all.backward()
            
            torch.nn.utils.clip_grad_norm_(self.reweighter.parameters(),2.0)
            self.reweighter_opt.step()

        #-----------------
        #show performance
        #-----------------
        #ipdb.set_trace()
        #utils.print_class_acc(output, data.y)
        #roc_train, macroF_train = utils.Roc_F(output, data.y)
        '''
        print('Epoch: {:05d}'.format(epoch+1),
              'loss_train ori: {:.4f}'.format(loss_train_ori.item()),
            'loss_train_re: {:.4f}'.format(loss_train_re.item()),
            'acc_train: {:.4f}'.format(acc_train.item()),
            'loss_weight: {:.4f}'.format(loss_weight.item()),
            'weight_reg: {:.4f}'.format(weight_reg.item()),
            'loss_reweight_cls: {:.4f}'.format(loss_reweight_cls.item())
            )
        '''
        log_info = {'loss_train_ori': loss_train_ori.item(), 'loss_train_reweighted': loss_train_re.item(), 'acc_train': acc_train.item(),
                     'loss_weight': loss_weight.item(), 'weight_reg': weight_reg.item(), 'loss_reweight_cls': loss_reweight_cls.item() }

        return log_info


    def train_mem_step(self, data, epoch):
        for i, model in enumerate(self.models):
            model.train()
            self.models_opt[i].zero_grad()
        self.reweighter.train()

        if self.args.shared_encoder:
            ipdb.set_trace()#not tested yet, for supporting batch information
            output = self.get_em(data)
            output = self.models[-1](output)
        else:
            output = self.models[0](data.x, data.edge_index, batch=data.batch)

        #-----------------
        #update reweighter
        #-----------------
        self.reweighter_mem_opt.zero_grad()
        all_weights = self.reweighter.predict_weight(data.x, data.edge_index,batch=data.batch, return_sim=False).squeeze()
        weights = all_weights
        with torch.no_grad():
            # loss_weight_back = -F.nll_loss(output.detach(), data.y, reduction='none')
            loss_weight_back = F.nll_loss(output.detach(), data.y, reduction='none')
        loss_weight = torch.mean(weights*(loss_weight_back.detach()))
        weight_reg = loss_weight_back.detach().mean().abs()*weights.mean()
        #loss_weight_all = loss_weight+weight_reg

        loss_weight_all = loss_weight


        re_output = self.reweighter(data.x, data.edge_index, batch=data.batch)
        # loss_reweight_cls = F.nll_loss(re_output, data.y)
        loss_reweight_cls = -F.nll_loss(re_output, data.y)
        loss_weight_all = loss_weight_all+loss_reweight_cls

        loss_weight_all = loss_weight_all
        
        loss_weight_all.backward()
        
        torch.nn.utils.clip_grad_norm_(self.reweighter.parameters(),2.0)
        self.reweighter_mem_opt.step()
        

        #-----------------
        #show performance
        #-----------------
        #ipdb.set_trace()
        #utils.print_class_acc(output, data.y)
        #roc_train, macroF_train = utils.Roc_F(output, data.y)
        '''
        print('Epoch: {:05d}'.format(epoch+1),
            'loss_weight_mem: {:.4f}'.format(loss_weight.item()),
            'weight_reg_mem: {:.4f}'.format(weight_reg.item()),
            'loss_reweight_cls_mem: {:.4f}'.format(loss_reweight_cls.item())
            )
        '''
        log_info = {'loss_weight_mem': loss_weight.item(), 'weight_reg_mem': weight_reg.item(), 'loss_reweight_cls_mem': loss_reweight_cls.item() }

        return log_info

    def test(self, data_loader, device, use_print=True):
        for i, model in enumerate(self.models):
            model.eval()
        self.reweighter.eval()

        all_predictions = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch, data in enumerate(data_loader):
                data = data.to(device)
                output = self.models[0](data.x, data.edge_index, batch=data.batch)
                preds = output.max(1)[1].type_as(data.y)
                all_predictions.append(preds.cpu().numpy())
                all_labels.append(data.y.cpu().numpy())
                probs = torch.softmax(output, dim=1)
                # 将预测概率转换为二维数组，并添加到all_probs中
                probs_array = probs.detach().cpu().numpy()
                if len(probs_array.shape) == 1:
                    probs_array = probs_array.reshape(-1, 1)
                all_probs.append(probs_array)

        all_predictions = np.concatenate(all_predictions, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_probs = np.concatenate(all_probs, axis=0)

        acc_test = accuracy_score(all_labels, all_predictions)
        bacc_test = balanced_accuracy_score(all_labels, all_predictions)

        if all_labels.max() > 1:  # require set(labels) to be the same as columns of logits
            auc_score = roc_auc_score(all_labels, all_probs, multi_class='ovr')
        else:
            auc_score = roc_auc_score(all_labels, all_probs[:, 1])

        macro_F = f1_score(all_labels, all_predictions, average='macro')
        log_info = {'acc_test': acc_test, 'roc_test': auc_score, 'macroF_test': macro_F,
                    'bacc_test': bacc_test}

        return log_info

    def calculate_silhouette(self, data_loader, device, data_name, seed):
        for i, model in enumerate(self.models):
            model.eval()
        self.reweighter.eval()
        all_embeddings = []
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for batch, data in enumerate(data_loader):
                data = data.to(device)
                output = self.models[0](data.x, data.edge_index, batch=data.batch)
                preds = output.max(1)[1].type_as(data.y)
                all_embeddings.append(output.cpu().numpy())
                all_predictions.append(preds.cpu().numpy())
                all_labels.append(data.y.cpu().numpy())

        # 将嵌入、预测和标签合并
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        print('All Predictions and All Labels')
        print(all_predictions)
        print(all_labels)
        # 分类别保存嵌入
        unique_classes = np.unique(all_predictions)
        for cls in unique_classes:
            cls_embeddings = all_embeddings[all_predictions == cls]
            os.makedirs(f'embeddings/{data_name}/', exist_ok=True)
            np.save(f'embeddings/{data_name}/class_{cls}_{seed}.npy', cls_embeddings)

        # 计算轮廓系数
        silhouette_avg = silhouette_score(all_embeddings, all_labels)
        return silhouette_avg

    def test_all_topo(self, data, topo_label, only_test=False, only_train=False, return_weights=False, return_sims=False):
        #both data and topo_label should be in torch.tensor
        #only_test = True: examine only test data; only_test = False: examine all data
        for i, model in enumerate(self.models):
            model.eval()
        self.reweighter.eval()

        if self.args.shared_encoder:
            ipdb.set_trace()#not tested for batch yet
            output = self.get_em(data)
            output = self.models[-1](output)
        else:
            output = self.models[0](data.x, data.edge_index, batch=data.batch)

        all_weights, sim_list = self.reweighter.predict_weight(data.x, data.edge_index, batch=data.batch, return_sim=True)
        all_weights = all_weights.squeeze()

        topo_sim_dict_list = []#show topo-group-wise memory cell selection behavior
        
        topo_ac_dict = utils.grouped_accuracy(output.cpu().detach().numpy(), data.y.cpu().numpy(), topo_label.cpu().numpy())
        topo_weight_dict = utils.grouped_measure(all_weights.cpu().detach().numpy(), topo_label.cpu().numpy())
        for sim in sim_list:
            topo_sim_dict = utils.grouped_measure(sim.cpu().detach().numpy(), topo_label.cpu().numpy())
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
