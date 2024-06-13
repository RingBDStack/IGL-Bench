import torch
import torch.nn.functional as F
import torch.nn as nn

from config import get_config,dirs
from OurLog import OurLog
from dataloader import load_processed_data
from utils import set_seed,index2adj_bool,save_results
import sys
import os
import numpy as np
import copy
from sklearn.metrics import f1_score,roc_auc_score
import json
import pickle
from losses.GraphAUC import GAUCLoss,ELossFN
from losses.imb_loss import IMB_LOSS
from sklearn.metrics import f1_score, accuracy_score,balanced_accuracy_score
from sklearn.preprocessing import label_binarize


def update_parameters_from_best_json(opts,tuner_params):
    opts.lr=tuner_params['lr']['_value'][0]
    opts.lr_decay_rate=tuner_params['lr_decay_rate']['_value'][0]
    opts.dropout=tuner_params['dropout']['_value'][0]
    opts.num_hidden=tuner_params['num_hidden']['_value'][0]
    opts.num_layer=tuner_params['num_layer']['_value'][0]
    if "pagerank_prob" in tuner_params.keys():
        opts.pagerank_prob=tuner_params['pagerank_prob']['_value'][0]
    if "weight_decay" in tuner_params.keys():
        opts.weight_decay=tuner_params['weight_decay']['_value'][0]
    if "weight_sub_dim" in tuner_params.keys():
        opts.weight_sub_dim=tuner_params['weight_sub_dim']['_value'][0]
    if "weight_inter_dim" in tuner_params.keys():
        opts.weight_inter_dim=tuner_params['weight_inter_dim']['_value'][0]
    if "weight_global_dim" in tuner_params.keys():
        opts.weight_global_dim=tuner_params['weight_global_dim']['_value'][0]
    if "beta" in tuner_params.keys():
        opts.beta=tuner_params['beta']['_value'][0]
    if "gamma" in tuner_params.keys():
        opts.gamma=tuner_params['gamma']['_value'][0]
    if "warm_up_epoch" in tuner_params.keys():
        opts.warm_up_epoch=tuner_params['warm_up_epoch']['_value'][0]
    if "warm_up_loss" in tuner_params.keys():
        opts.warm_up_loss=tuner_params['warm_up_loss']['_value'][0]
    if "rn_base_weight" in tuner_params.keys():
        opts.rn_base_weight=tuner_params['rn_base_weight']['_value'][0]
    if "rn_scale_weight" in tuner_params.keys():
        opts.rn_base_weight=tuner_params['rn_scale_weight']['_value'][0]
    opts.gem_file = "{}/{}_{:.2f}_gem.pt".format(dirs["DATA_PATH"],opts.dataset,opts.pagerank_prob) # the pre-computed global effect matrix
    opts.saved_model="saved_models/{}_{}_{}_is_renode_{}_best_model.pt".format(opts.dataset,opts.model,opts.loss,opts.renode_reweight)
    return opts

def get_model(opts):
    nfeat = opts.num_feature
    nclass = opts.num_class
    nhid = opts.num_hidden
    nlayer = opts.num_layer
        
    dropout = opts.dropout
    model_opt = opts.model

    from models.GCN import StandGCN1,StandGCN2,StandGCNX
    from models.gat import StandGAT1,StandGAT2,StandGATX
    from models.cheb_gcn import ChebGCN1,ChebGCN2,ChebGCNX
    from models.sage import GraphSAGE1,GraphSAGE2,GraphSAGEX
    from models.ppnp import PPNP1,PPNP2,PPNPX
    from models.sgc import SGC1,SGC2,SGCX
    model_dict = {
        'gcn'  : [StandGCN1,StandGCN2,StandGCNX],
        'gat'  : [StandGAT1,StandGAT2,StandGATX],
        'cheb' : [ChebGCN1,ChebGCN2,ChebGCNX],
        'sage' : [GraphSAGE1,GraphSAGE2,GraphSAGEX],
        'ppnp' : [PPNP1,PPNP2,PPNPX],
        'sgc'  : [SGC1,SGC2,SGCX],
    }
    model_list = model_dict[model_opt]
    if nlayer==1:
        model = model_list[0](nfeat=nfeat,
                nhid=nhid,
                nclass=nclass,
                dropout=dropout,
                nlayer = nlayer)

    elif nlayer ==2:
        model = model_list[1](nfeat=nfeat,
                nhid=nhid,
                nclass=nclass,
                dropout=dropout,
                nlayer = nlayer)

    else:
        model = model_list[2](nfeat=nfeat,
                nhid=nhid,
                nclass=nclass,
                dropout=dropout,
                nlayer = nlayer)  

    return model.to(opts.device)

def test(opts, model, data, adj, target_mask, test_type=''):
    model.eval()
    target = data.y[target_mask].numpy()

    with torch.no_grad():
        out = model(data.x.to(opts.device), adj.to(opts.device))
    
    soft_out = F.softmax(out[target_mask], dim=1)
    soft_out = soft_out.cpu().numpy()
    pred = out[target_mask].cpu().max(1)[1].numpy()

    w_f1 = f1_score(target, pred, average='weighted')
    m_f1 = f1_score(target, pred, average='macro')
    acc = accuracy_score(target, pred)
    bacc = balanced_accuracy_score(target, pred)

    n_classes = soft_out.shape[1]
    target_binarized = label_binarize(target, classes=range(n_classes))
    
    auroc = roc_auc_score(target_binarized, soft_out, multi_class='ovr', average='macro')

    return w_f1, m_f1, acc, bacc, auroc

def train(model,opts,data,edge_index,gem,log):
    #my_loss = IMB_LOSS(opt.loss_name,opt,data)
    adj_bool=index2adj_bool(edge_index,data.num_nodes).to(opts.device)
    if opts.loss in ["ce","focal","re-weight","cb-softmax"]:
        my_loss=IMB_LOSS(opts.loss,opts,data)
    else:
        my_loss=ELossFN(data.num_classes,data.num_nodes,adj_bool,gem,data.gpr,data.train_mask,opts.device,weight_sub_dim=opts.weight_sub_dim,weight_inter_dim=opts.weight_inter_dim,weight_global_dim=opts.weight_global_dim,beta= opts.beta,gamma=opts.gamma,is_ner_weight=opts.pair_ner_diff,loss_type=opts.loss)

    
    if opts.loss in ["ExpGAUC","HingeGAUC","SqGAUC"]:
        if opts.warm_up_epoch:
            my_warm_up_loss=IMB_LOSS(opts.warm_up_loss,opts,data)
    
    if opts.loss in ["ExpGAUC","HingeGAUC","SqGAUC"]:
        optimizer = torch.optim.Adam([*list(model.parameters()),*list(my_loss.parameters())], lr=opts.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=opts.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=opts.weight_decay)

    best_acc = 0
    best_w_f1=0
    best_m_f1=0
    best_epoch = 0
    best_auroc=0

    w_values_dict={}

    for epoch in range(1, opts.epoch+1):
        if epoch > opts.lr_decay_epoch:
            new_lr = opts.lr * pow(opts.lr_decay_rate,(epoch-opts.lr_decay_epoch))
            new_lr = max(new_lr,1e-4)
            for param_group in optimizer.param_groups: param_group['lr'] = new_lr

        model.train()
        total_loss = 0
        data.batch = None
        optimizer.zero_grad()

        if opts.loss in ["ce","focal","re-weight","cb-softmax"]:
            sup_logits  = model(data.x.to(opts.device), edge_index.to(opts.device))
            cls_loss    = my_loss.compute(sup_logits[data.train_mask], data.y[data.train_mask].to(opts.device))
            if opts.renode_reweight == 1:
                cls_loss = torch.sum(cls_loss * data.rn_weight[data.train_mask].to(opts.device)) / cls_loss.size(0)
            else:
                cls_loss = torch.mean(cls_loss)
            cls_loss.backward()
            optimizer.step()
        else:
            if opts.warm_up_epoch and epoch<=opts.warm_up_epoch:
                sup_logits  = model(data.x.to(opts.device), edge_index.to(opts.device))
                cls_loss    = my_warm_up_loss.compute(sup_logits[data.train_mask], data.y[data.train_mask].to(opts.device))
                cls_loss = torch.mean(cls_loss)
                cls_loss.backward()
                optimizer.step()
            else:
                sup_logits  = model(data.x.to(opts.device), edge_index.to(opts.device))
                sup_logits=F.softmax(sup_logits,dim=1)
                #cls_loss    = my_loss(sup_logits[data.train_mask], data.y[data.train_mask].to(opts.device),data.train_mask,w_values_dict)
                cls_loss = my_loss(sup_logits, data.y.to(opts.device),data.train_mask,w_values_dict)
                cls_loss = torch.mean(cls_loss)
                cls_loss.backward()
                optimizer.step()
        
        train_loss = cls_loss / data.train_mask.size(0)
        w_f1,m_f1,acc,bacc,auroc    = test(opts,model,data,edge_index,data.valid_mask)

        if acc>best_acc:
            best_model = copy.deepcopy(model)
            best_acc = acc
            best_w_f1=w_f1
            best_m_f1=m_f1
            best_epoch = epoch 
            best_auroc = auroc

        log.info('Epoch [{:02d}] | lr[{:.6f}] | Loss[{:.6f}] |  W-F[{:.4f}] | M-F[{:.4f}]'\
            .format(epoch,optimizer.param_groups[0]['lr'],cls_loss,w_f1,m_f1))
            
        
        test_w_f1,test_m_f1,test_acc,test_bacc,test_auroc = test(opts,best_model, data,edge_index,data.test_mask,'test')
        #log.info('Epoch [%.02d] | [test] W-F:[%.4f], M-F:[%.4f],AUC_OVO:[%.4f]'%(epoch,test_w_f1,test_m_f1,test_auc_ovo))

        if opts.nni:
            import nni
            nni.report_intermediate_result({'default':test_auc_ovo,'test_m_f1':test_m_f1,'test_w_f1':test_w_f1,'train_loss':cls_loss,'val_w_f1':w_f1,'val_m_f1':m_f1,'val_auc_ovo':auc_ovo})
        
        if opts.early_stop>0 and epoch>opts.least_epoch and epoch - best_epoch > opts.early_stop+opts.warm_up_epoch: 
            log.info('Early stop at %d epoch. Since there is no improve in %d epoch'%(epoch,opts.early_stop))
            break

    torch.save(best_model.state_dict(),opts.saved_model)

    #log.info('[val] best_epoch:[%d],W-F:[%.4f], M-F:[%.4f],best_AUC_OVO:[%.4f]'%(best_epoch,best_w_f1,best_m_f1,best_auc_ovo))
    #pickle.dump(w_values_dict,open('exp_results/'+str(opts.dataset)+'_'+opts.loss+'_imb_'+str(int(opts.imb_ratio))+'.pkl','wb'))

    del optimizer
    del my_loss

    return best_model

def main(opts):
    if opts.gpu>-1:
        opts.device = torch.device("cuda")
        torch.cuda.set_device(0)
    else:
        opts.device = torch.device("cpu")
    cur_log=OurLog("{}_{}_{}.log".format(opts.dataset,opts.model,opts.loss))
    cur_log.info(opts)

    run_time_result_weighted_f1 = [[] for _ in range(opts.run_split_num)]
    run_time_result_macro_f1    = [[] for _ in range(opts.run_split_num)]
    run_time_result_acc    = [[] for _ in range(opts.run_split_num)]
    run_time_result_bacc    = [[] for _ in range(opts.run_split_num)]
    run_time_result_auroc    = [[] for _ in range(opts.run_split_num)]

    for iter_split_seed in range(opts.run_split_num):
        cur_log.info('The [%d] / [%d] dataset spliting...'%(iter_split_seed+1,opts.run_split_num))
        cur_log.info('Loading data...') 

        target_data = load_processed_data(opts,cur_log,dirs["DATA_PATH"],opts.dataset,shuffle_seed = opts.shuffle_seed,gem_file = opts.gem_file)
        adj = target_data.edge_index
        gem=target_data.Pi
        cur_log.info(target_data)



        setattr(opts, 'num_feature', target_data.num_features)
        setattr(opts, 'num_class', target_data.num_classes)
        for iter_init_seed in range(opts.run_init_num):
            set_seed((opts.seed_list[iter_init_seed])+1,True)

            model = get_model(opts)
            cur_log.info('Training begining...')
            best_model = train(model,opts,target_data,adj,gem,cur_log)
            cur_log.info('Testing begining...')
            w_f1,m_f1,acc,bacc,auroc  = test(opts,best_model, target_data,adj,target_data.test_mask,'test')
            cur_log.info('[test] W-F:[%.4f], M-F:[%.4f],ACC:[%.4f],bACC:[%.4f]'%(w_f1,m_f1,acc,bacc))
            
            del model
            del best_model


    #if opts.save_repeated_res:
        #save_results(run_time_result_weighted_f1,run_time_result_macro_f1,run_time_result_macro_auc_ovo,opts.dataset,opts.loss+'_renode_'+str(opts.renode_reweight)+'_pntd_'+str(opts.pair_ner_diff)+'_imb_'+str(int(opts.imb_ratio))+'_gnnLayer_'+str(opts.num_layer)+'.csv')
    #cur_log.info('The overall performance:')

    weighted_f1_np   = np.array(run_time_result_weighted_f1)
    weighted_f1_mean = np.mean(weighted_f1_np)
    weighted_f1_std  = np.std(weighted_f1_np)
    #weighted_f1_max=np.max(weighted_f1_np)

    macro_f1_np = np.array(run_time_result_macro_f1)
    macro_f1_mean = np.mean(macro_f1_np)
    macro_f1_std  = np.std(macro_f1_np)
    #macro_f1_max=np.max(macro_f1_np)

    acc_np = np.array(run_time_result_acc)
    acc_mean = np.mean(acc_np )
    acc_std  = np.std(acc_np )
    #acc_max=np.max(acc_np )
    
    bacc_np = np.array(run_time_result_bacc)
    bacc_mean = np.mean(bacc_np )
    bacc_std  = np.std(bacc_np )   
    
    auroc_np = np.array(run_time_result_auroc)
    auroc_mean = np.mean(auroc_np )
    auroc_std  = np.std(auroc_np ) 
    
    print("Weighted_F1 | Macro_F1 | ACC | bACC | AUROC\n{:.2f}-{:.2f}   {:.2f}-{:.2f}   {:.2f}-{:.2f}   {:.2f}-{:.2f}   {:.2f}-{:.2f}".format(100*weighted_f1_mean,100*weighted_f1_std,
    100*macro_f1_mean,100*macro_f1_std,100*acc_mean,100*acc_std,100*bacc_mean,100*bacc_std,100*auroc_mean,100*auroc_std))
    directory = "res"
    filename = f"{opts.dataset}_{opts.model}.txt"
    filepath = os.path.join(directory, filename)
    os.makedirs(directory, exist_ok=True)
    with open(filepath, "a") as file:
        file.write(f"seed : {opts.shuffle_seed}\n")
        file.write("Weighted_F1 | Macro_F1 | ACC | bACC | AUROC\n")
        file.write("{:.2f}-{:.2f}   {:.2f}-{:.2f}   {:.2f}-{:.2f}   {:.2f}-{:.2f}   {:.2f}-{:.2f}\n".format(
            100*weighted_f1_mean, 100*weighted_f1_std,
            100*macro_f1_mean, 100*macro_f1_std,
            100*acc_mean, 100*acc_std,
            100*bacc_mean, 100*bacc_std,
            100*auroc_mean, 100*auroc_std))
    
    #cur_log.info(opts)
    #cur_log.info("Weighted_F1: {:.2f}±{:.2f},max:{:.2f} | Macro_F1: {:.2f}±{:.2f},max:{:.2f} | Macro_AUC_OVO: {:.2f}±{:.2f},max:{:.2f}".format(100*weighted_f1_mean,100*weighted_f1_std,100*weighted_f1_max,100*macro_f1_mean,100*macro_f1_std,100*macro_f1_max,100*macro_auc_ovo_mean,100*macro_auc_ovo_std,100*macro_auc_ovo_max))




def get_tot():
    opts = get_config()
    opts.model='gcn'
    opts.num_layer=3

    imbs=['10','15','20']
    imbs_map={'10':10.0,'15':15.0,'20':20.0}
    loss0=['ce']
    loss1=['cb','focal','rw']
    loss2=['expgauc','hingegauc','sqgauc']
    loss_map={'ce':'ce','cb':"cb-softmax",'focal':"focal",'rw':"re-weight",'expgauc':"ExpGAUC",'hingegauc':"HingeGAUC",'sqgauc':"SqGAUC"}
    datasets=["cora","citeseer"]
    for cur_dataset in datasets:
        pre_dir=os.path.join('best_params','layers3',cur_dataset)
        opts.dataset=cur_dataset
        for cur_imb in imbs:
            opts.imb_ratio=imbs_map[cur_imb]
            for cur_loss in loss0:
                params_file=os.path.join(pre_dir,'search_space_imb_losses_'+cur_loss+'_imb_'+cur_imb+'.json')
                opts.loss=loss_map[cur_loss]
                opts.pair_ner_diff=0
                opts.renode_reweight=0
                params=json.load(open(params_file,'r'))
                opts=update_parameters_from_best_json(opts,params)
                opts.gem_file = "{}/{}_gem.pt".format(dirs["DATA_PATH"],opts.dataset) # the pre-computed global effect matrix
                opts.saved_model="saved_models/{}_{}_{}_is_renode_{}_is_pntd_{}_best_model.pt".format(opts.dataset,opts.model,opts.loss,opts.renode_reweight,opts.pair_ner_diff)
                main(opts)
            for cur_loss in loss1:
                params_file=os.path.join(pre_dir,'search_space_imb_losses_'+cur_loss+'_renode_'+cur_imb+'.json')
                opts.loss=loss_map[cur_loss]
                opts.pair_ner_diff=0
                opts.renode_reweight=0
                params=json.load(open(params_file,'r'))
                opts=update_parameters_from_best_json(opts,params)
                opts.gem_file = "{}/{}_gem.pt".format(dirs["DATA_PATH"],opts.dataset) # the pre-computed global effect matrix
                opts.saved_model="saved_models/{}_{}_{}_is_renode_{}_is_pntd_{}_best_model.pt".format(opts.dataset,opts.model,opts.loss,opts.renode_reweight,opts.pair_ner_diff)
                main(opts)
                opts.renode_reweight=1
                opts.gem_file = "{}/{}_gem.pt".format(dirs["DATA_PATH"],opts.dataset) # the pre-computed global effect matrix
                opts.saved_model="saved_models/{}_{}_{}_is_renode_{}_is_pntd_{}_best_model.pt".format(opts.dataset,opts.model,opts.loss,opts.renode_reweight,opts.pair_ner_diff)
                main(opts)
            for cur_loss in loss2:
                params_file=os.path.join(pre_dir,'search_space_imb_losses_'+cur_loss+'_imb_'+cur_imb+'.json')
                opts.loss=loss_map[cur_loss]
                opts.renode_reweight=0
                opts.pair_ner_diff=0
                params=json.load(open(params_file,'r'))
                opts=update_parameters_from_best_json(opts,params)
                opts.gem_file = "{}/{}_gem.pt".format(dirs["DATA_PATH"],opts.dataset) # the pre-computed global effect matrix
                opts.saved_model="saved_models/{}_{}_{}_is_renode_{}_is_pntd_{}_best_model.pt".format(opts.dataset,opts.model,opts.loss,opts.renode_reweight,opts.pair_ner_diff)
                main(opts)
                opts.pair_ner_diff=1
                opts.gem_file = "{}/{}_gem.pt".format(dirs["DATA_PATH"],opts.dataset) # the pre-computed global effect matrix
                opts.saved_model="saved_models/{}_{}_{}_is_renode_{}_is_pntd_{}_best_model.pt".format(opts.dataset,opts.model,opts.loss,opts.renode_reweight,opts.pair_ner_diff)
                main(opts)

if __name__ == '__main__':
    # get_tot()
    opts = get_config()
    
    #["ce","focal","re-weight","cb-softmax","ExpGAUC","HingeGAUC","SqGAUC"]
    #opts.dataset="cora" #["cora","citeseer","pubmed","photo","computers"]
    #opts.loss='SqGAUC'

    #opts.pair_ner_diff=1#add topology weight
    #opts.renode_reweight=0 #1 for other losses(not our TOPOAUC)
    #opts.imb_ratio=20.0#10,15,20
    #opts.model='gcn' ##gcn gat ppnp sage cheb sgc

    #opts.gem_file = "{}/{}_gem.pt".format(dirs["DATA_PATH"],opts.dataset) # the pre-computed global effect matrix
    #opts.saved_model="saved_models/{}_{}_{}_is_renode_{}_best_model.pt".format(opts.dataset,opts.model,opts.loss,opts.renode_reweight)
    #opts.weight_decay=4e-3
    #best_params_json_file='best_params/layers3/cora/search_space_imb_losses_sqgauc_imb_20.json'
    #if os.path.exists(best_params_json_file):
    #    params=json.load(open(best_params_json_file,'r'))
    #    opts=update_parameters_from_best_json(opts,params)
    main(opts)