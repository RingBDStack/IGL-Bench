import os
import random
import sys,time
import copy
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from scipy.sparse import coo_matrix

from load_data import load_processed_data
from models import get_model
from utils import set_seed,log_opt
from imb_loss import IMB_LOSS
from opts import get_opt,save_to_yaml
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize

import warnings
warnings.simplefilter("ignore")



def train(model,opt,data,adj,log_writer = None):
    my_loss = IMB_LOSS(opt.loss_name,opt,data)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=opt.weight_decay)

    best_res = 0
    best_epoch = 0

    for epoch in range(1, opt.epoch+1):
        if epoch > opt.lr_decay_epoch:
            new_lr = opt.lr * pow(opt.lr_decay_rate,(epoch-opt.lr_decay_epoch))
            new_lr = max(new_lr,1e-4)
            for param_group in optimizer.param_groups: param_group['lr'] = new_lr

        model.train()
        total_loss = 0
        data.batch = None
        optimizer.zero_grad()

        sup_logits  = model(data.x.to(opt.device), adj.to(opt.device))        
        cls_loss    = my_loss.compute(sup_logits[data.train_mask], data.y[data.train_mask].to(opt.device))

        if opt.renode_reweight == 1:
            cls_loss = torch.sum(cls_loss * data.rn_weight[data.train_mask].to(opt.device)) / cls_loss.size(0)
        else:
            cls_loss = torch.mean(cls_loss)
        
        cls_loss.backward()
        optimizer.step()
        
        train_loss = cls_loss / data.train_mask.size(0)
        val_res    = test(opt,model,data,adj,data.valid_mask)

        if val_res>best_res:
            best_model = copy.deepcopy(model)
            best_res = val_res
            best_epoch = epoch 

        sys.stdout.write('\rEpoch[{:02d}] | lr[{:.4f}] | SuperL[{:.4f}] |  W-F[{:.4f}] '\
            .format(epoch,optimizer.param_groups[0]['lr'],cls_loss,val_res))
                
        log_writer.write('Epoch[{:02d}] | lr[{:.4f}] | SuperL[{:.4f}] | W-F[{:.4f}]\n'\
            .format(epoch,optimizer.param_groups[0]['lr'],cls_loss,val_res))
            
        if opt.early_stop>0 and epoch>opt.least_epoch and epoch - best_epoch > opt.early_stop: 
            print('\nEarly stop at %d epoch. Since there is no improve in %d epoch'%(epoch,opt.early_stop))
            break

    torch.save(best_model.state_dict(),opt.saved_model)

    print('best_epoch,best_val_result:%d, %.4f'%(best_epoch,best_res))
    log_writer.write('best_epoch,best_val_result:%d, %.4f\n'%(best_epoch,best_res))

    del optimizer
    del my_loss

    return best_model

def test(opt, model, data, adj, target_mask, test_type=''):
    model.eval()
    target = data.y[target_mask].numpy()

    with torch.no_grad():
        out = model(data.x.to(opt.device), adj.to(opt.device))
    pred = out[target_mask].cpu().max(1)[1].numpy()
    pred_prob = out[target_mask].cpu().numpy()  

    w_f1 = f1_score(target, pred, average='weighted')
    
    if test_type == 'test':
        m_f1 = f1_score(target, pred, average='macro')
        acc = accuracy_score(target, pred)
        bacc = balanced_accuracy_score(target, pred)

        n_classes = data.y.max().item() + 1  
        target_binarized = label_binarize(target, classes=range(n_classes))

        roc_auc = roc_auc_score(target_binarized, pred_prob, multi_class='ovr')
        
        return w_f1, m_f1, acc, bacc, roc_auc

    return w_f1


def main(opt):

    # setting env
    print('\nSetting environment...')
    if opt.gpu>-1:
        opt.device = torch.device("cuda:{}".format(opt.gpu))
    else:
        opt.device = torch.device("cpu")

    log_writer = open(opt.log_path,'w')
    log_opt(opt,log_writer)

    run_time_result_weighted = [[] for _ in range(opt.run_split_num)]
    run_time_result_macro    = [[] for _ in range(opt.run_split_num)]
    run_time_result_acc    = [[] for _ in range(opt.run_split_num)]
    run_time_result_bacc    = [[] for _ in range(opt.run_split_num)]
    run_time_result_auroc    = [[] for _ in range(opt.run_split_num)]


    for iter_split_seed in range(opt.run_split_num):
        print('The [%d] / [%d] dataset spliting...'%(iter_split_seed+1,opt.run_split_num))

        print('\nLoading data...')
        target_data = load_processed_data(opt,opt.data_path,opt.data_name,
            shuffle_seed = opt.shuffle_seed,
            ppr_file = opt.ppr_file)
        setattr(opt, 'num_feature', target_data.num_features)
        setattr(opt, 'num_class', target_data.num_classes)

        adj = target_data.edge_index
        
        for iter_init_seed in range(opt.run_init_num):
            print('--[%d] / [%d] seed...'%(iter_init_seed+1,opt.run_init_num))
            log_writer.write('\n--[%d] / [%d] seed:\n'%(iter_init_seed+1,opt.run_init_num))
            
            #set the seed for training initial
            set_seed(opt.seed_list[iter_init_seed], opt.gpu>-1)

            print('\nSetting model...')
            model = get_model(opt)

            if opt.model in ['ppnp'] : model.norm_adj(opt,target_data)

            if iter_split_seed == 0 and iter_init_seed == 0 : print(model)
 
            print('\nTraining begining...')
            best_model = train(model,opt,target_data,adj,log_writer)

            print('\nTesting begining..')
            weighted_f1,macro_f1,acc,bacc,auroc  = test(opt,best_model, target_data,adj,target_data.test_mask,'test')

            print("Weighted_F1 | Macro_F1 \n {:.4f};{:.4f}".format(weighted_f1,macro_f1))

            run_time_result_weighted[iter_split_seed].append(weighted_f1)
            run_time_result_macro[iter_split_seed].append(macro_f1)
            run_time_result_acc[iter_split_seed].append(acc)
            run_time_result_bacc[iter_split_seed].append(bacc)
            run_time_result_auroc[iter_split_seed].append(auroc)

            del model
            del best_model

    print('\nThe overall performance:')
    weighted_np   = np.array(run_time_result_weighted)
    weighted_mean = round(np.mean(weighted_np),4)
    weighted_std  = round(np.std(weighted_np),4)

    macro_np = np.array(run_time_result_macro)
    macro_mean = round(np.mean(macro_np),4)
    macro_std  = round(np.std(macro_np),4)

    acc_np = np.array(run_time_result_acc)
    acc_mean = round(np.mean(acc_np),4)
    acc_std  = round(np.std(acc_np),4)
    
    bacc_np = np.array(run_time_result_bacc)
    bacc_mean = round(np.mean(bacc_np),4)
    bacc_std  = round(np.std(bacc_np),4)    
    
    auroc_np = np.array(run_time_result_auroc)
    auroc_mean = round(np.mean(auroc_np),4)
    auroc_std  = round(np.std(auroc_np),4)   

    print("Weighted_F1 | Macro_F1 | ACC | bACC | AUROC\n{:.3f}-{:.3f}   {:.3f}-{:.3f}   {:.3f}-{:.3f}   {:.3f}-{:.3f} {:.3f}-{:.3f}".format(100*weighted_mean,100*weighted_std,100*macro_mean,100*macro_std,100*acc_mean,\
        100*acc_std,100*bacc_mean,100*bacc_std,100*auroc_mean,100*auroc_std))
    log_writer.write("\n\nWeighted_F1 | Macro_F1 \n{:.2f}-{:.2f}\n{:.2f}-{:.2f}\n".format(100*weighted_mean,100*weighted_std,100*macro_mean,100*macro_std))
    log_writer.close()
    file_name = f"res/{opt.data_name}.txt"
    with open(file_name, "a") as file:
        file.write(f"seed : {opt.shuffle_seed}\n")
        file.write(" ACC | bACC | Macro_F1 |  AUROC\n")
        file.write("{:.2f}-{:.2f}   {:.2f}-{:.2f}   {:.2f}-{:.2f}   {:.2f}-{:.2f}\n".format(
        100*acc_mean, 100*acc_std, 100*bacc_mean, 100*bacc_std,100*macro_mean, 100*macro_std,100*auroc_mean,100*auroc_std))


# -----------------------------------------------
if __name__ == '__main__':
    opt = get_opt()   
    save_to_yaml(opt) 
    main(opt)
