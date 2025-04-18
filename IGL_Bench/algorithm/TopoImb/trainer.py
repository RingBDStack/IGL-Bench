import torch
import torch.optim as optim
import torch.nn.functional as F

import IGL_Bench.algorithm.TopoImb.utils as utils 
from IGL_Bench.algorithm.TopoImb.layers import MLP
from sklearn.metrics import balanced_accuracy_score

def cal_feat_dim(args): 

    emb_dim = args.hidden_dim
    emb_dim = emb_dim * 2
    emb_dim = emb_dim*args.n_layer

    return emb_dim

class Trainer(object):
    def __init__(self, config, model, weight):
        self.config = config
        self.loss_weight = weight
        self.models = [model]

        self.models_opt = []
        for model in self.models:
            self.models_opt.append(optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay))

    def train_step(self, data):

        raise NotImplementedError('train not implemented for base class')

    def inference(self, data):
        raise NotImplementedError('infer not implemented for base class')

    def get_em(self, data):
        output = self.models[0].embedding(data.x.float(), data.edge_index)
        return output


class GClsTrainer(Trainer):
    def __init__(self, config, model, weight=1.0, dataset=None):
        super().__init__(config, model, weight)
        self.config = config

    def train_step(self, data, epoch):
        for i, model in enumerate(self.models):
            model.train()
            self.models_opt[i].zero_grad()

        output = self.models[0](data.x.float(), data.edge_index, batch=data.batch)

        loss_train = F.nll_loss(output, data.y)
        acc_train = utils.accuracy(output, data.y)

        loss_all = loss_train * self.loss_weight
        loss_all.backward()

        for model in self.models:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)

        for opt in self.models_opt:
            opt.step()

        print(
            f"Epoch: {epoch + 1:05d} "
            f"loss_train: {loss_train.item():.4f} "
            f"acc_train: {acc_train.item():.4f}"
        )

        log_info = {
            'loss_train': loss_train.item(),
            'acc_train': acc_train.item()
        }
        return log_info

    def test(self, data):
        for model in self.models:
            model.eval()

        output = self.models[0](data.x.float(), data.edge_index, batch=data.batch)

        loss_test = F.nll_loss(output, data.y)
        acc_test = utils.accuracy(output, data.y)

        print(
            "Test set results: "
            f"loss= {loss_test.item():.4f} "
            f"accuracy= {acc_test.item():.4f}"
        )

        utils.print_class_acc(output, data.y, pre='test')

        roc_test, macroF_test = utils.Roc_F(output, data.y, pre='test')
        log_info = {
            'loss_test': loss_test.item(),
            'acc_test': acc_test.item(),
            'roc_test': roc_test,
            'macroF_test': macroF_test
        }
        return log_info

    def test_all_topo(self, data, topo_label, only_test=False):
        for model in self.models:
            model.eval()

        output = self.models[0](data.x.float(), data.edge_index)

        if only_test:
            topo_ac_dict = utils.grouped_accuracy(
                output.cpu().detach().numpy(),
                data.y.cpu().numpy(),
                topo_label.cpu().numpy()
            )
        else:
            topo_ac_dict = utils.grouped_accuracy(
                output.cpu().detach().numpy(),
                data.y.cpu().numpy(),
                topo_label.cpu().numpy()
            )
        return topo_ac_dict

    def get_em(self, data):
        output = self.models[0].embedding(
            data.x.float(),
            data.edge_index,
            batch=data.batch,
            return_list=False,
            graph_level=True
        )
        return output

class ReweighterGraphTrainer(Trainer):
    def __init__(self, args, model, reweighter, weight=1.0, dataset=None, weights=None):
        super().__init__(args, model, weight)        

        self.args = args

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
            output = self.get_em(data)
            output = self.models[-1](output)
        else:
            output = F.log_softmax(self.models[0](data.x, data.edge_index, batch=data.batch),dim=-1)

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
        # print('Epoch: {:05d}'.format(epoch+1),
        #       'loss_train: {:.4f}'.format(loss_train_re.item()),
        #     'acc_train: {:.4f}'.format(acc_train.item()))
        
        log_info = {'loss_train_ori': loss_train_ori.item(), 'loss_train_reweighted': loss_train_re.item(), 'acc_train': acc_train.item(),
                     'loss_weight': loss_weight.item(), 'weight_reg': weight_reg.item(), 'loss_reweight_cls': loss_reweight_cls.item() }

        return log_info


    def train_mem_step(self, data, epoch):
        for i, model in enumerate(self.models):
            model.train()
            self.models_opt[i].zero_grad()
        self.reweighter.train()

        output = F.log_softmax(self.models[0](data.x, data.edge_index, batch=data.batch),dim=-1)

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

    def test(self, data, use_print=False):
        for i, model in enumerate(self.models):
            model.eval()
        self.reweighter.eval()

        if self.args.shared_encoder:
            output = self.get_em(data)
            output = self.models[-1](output)
        else:
            output = F.log_softmax(self.models[0](data.x, data.edge_index, batch=data.batch),dim=-1)

        output_prob = F.softmax(output, dim=1)

        loss_test = F.nll_loss(output, data.y)
        acc_test = utils.accuracy(output, data.y)

        bacc_test = balanced_accuracy_score(data.y.cpu().numpy(), output_prob.max(1)[1].cpu().detach().numpy()) 

        if use_print:
            print("Test set results:",
                "loss= {:.4f}".format(loss_test.item()),
                "accuracy= {:.4f}".format(acc_test.item()),
                "balanced accuracy= {:.2f}%".format(bacc_test))

            utils.print_class_acc(output, data.y, pre='test')
        
        roc_test, macroF_test = utils.Roc_F(output, data.y, pre='test')
        if 'topo_label' in data:
            topo_macro_acc, topo_macro_auc, topo_macro_F = utils.groupewise_perform(output, data.y, data.topo_label, pre='test')        
            log_info = {'loss_test': loss_test.item(), 'acc_test': acc_test.item(), 'roc_test': roc_test, 'macroF_test': macroF_test, 
                        'topo_macro_acc_test': topo_macro_acc, 'topo_macro_auc_test': topo_macro_auc, 'topo_macro_F_test': topo_macro_F,
                        'bacc_test': bacc_test}
        else:
            log_info = {'loss_test': loss_test.item(), 'acc_test': acc_test.item(), 'roc_test': roc_test, 'macroF_test': macroF_test,
                        'bacc_test': bacc_test}

        return log_info

    def test_all_topo(self, data, topo_label, only_test=False, only_train=False, return_weights=False, return_sims=False):
        #both data and topo_label should be in torch.tensor
        #only_test = True: examine only test data; only_test = False: examine all data
        for i, model in enumerate(self.models):
            model.eval()
        self.reweighter.eval()

        output = F.log_softmax(self.models[0](data.x, data.edge_index, batch=data.batch),dim=-1)

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
            
class TopoGraphTrainer(Trainer):
    '''for node classification
    
    '''
    def __init__(self, args, model, weight=1.0, dataset=None, out_size=None,shared_encoder=True):
        super().__init__(args, model, weight)        

        self.args = args
        self.shared_encoder = shared_encoder
        if shared_encoder:#provided model is the shared feature encoder, and trainer-specific classifier is required for each task
            self.in_dim = cal_feat_dim(args)
            if out_size is None:
                out_size = dataset[0].y.max().item()+1
            self.classifier = MLP(in_feat=self.in_dim, hidden_size=args.hidden_dim, out_size=out_size, layers=args.n_layer)
            self.classifier.cuda()
            self.classifier_opt = optim.Adam(self.classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
            self.models.append(self.classifier)
            self.models_opt.append(self.classifier_opt)


    def train_step(self, data, epoch, label=None):
        for i, model in enumerate(self.models):
            model.train()
            self.models_opt[i].zero_grad()

        if label is None:
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