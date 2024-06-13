import numpy as np
import torch
import random
import time
import copy
import os
import pandas as pd
#import tensorflow as tf

def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    #random.seed(seed)
    if cuda: torch.cuda.manual_seed(seed)

def index2dense(edge_index,nnode=2708):

    indx = edge_index.numpy()
    adj = np.zeros((nnode,nnode),dtype = 'int8')
    adj[(indx[0],indx[1])]=1
    new_adj = torch.from_numpy(adj).float()
    return new_adj
    
def index2adj_bool(edge_index,nnode=2708):

    indx = edge_index.numpy()
    adj = np.zeros((nnode,nnode),dtype = 'bool')
    adj[(indx[0],indx[1])]=1
    new_adj = torch.from_numpy(adj)
    return new_adj

def index2adj(inf,nnode = 2708):

    indx = inf.numpy()
    print(nnode)
    adj = np.zeros((nnode,nnode),dtype = 'int8')
    adj[(indx[0],indx[1])]=1
    return adj

def adj2index(inf):

    where_new = np.where(inf>0)
    new_edge = [where_new[0],where_new[1]]
    new_edge_tensor = torch.from_numpy(np.array(new_edge))
    return new_edge_tensor

def log_opt(opt,log_writer):
    for arg in vars(opt): log_writer.write("{}:{}\n".format(arg,getattr(opt,arg)))

def to_inverse(in_list,t=1):

    in_arr = np.array(in_list)
    in_mean = np.mean(in_arr)
    out_arr = in_mean / in_arr
    out_arr = np.power(out_arr,t)

    return out_arr

def save_results(run_time_result_weighted_f1,run_time_result_macro_f1,run_time_result_macro_auc_ovo,dataset,file_name):
    pre_dir=os.path.join('exp_results',dataset)
    if not os.path.exists(pre_dir):
        os.makedirs(pre_dir)
    save_path=os.path.join(pre_dir,file_name)
    new_w_f1,new_m_f1,new_m_auc=[],[],[]
    for i in range(len(run_time_result_weighted_f1)):
        for j in range(len(run_time_result_weighted_f1[i])):
            new_w_f1.append(run_time_result_weighted_f1[i][j])
    for i in range(len(run_time_result_macro_f1)):
        for j in range(len(run_time_result_macro_f1[i])):
            new_m_f1.append(run_time_result_macro_f1[i][j])
    for i in range(len(run_time_result_macro_auc_ovo)):
        for j in range(len(run_time_result_macro_auc_ovo[i])):
            new_m_auc.append(run_time_result_macro_auc_ovo[i][j])
    df_dict={'W-F1':new_w_f1,'M-F1':new_m_f1,'AUC':new_m_auc}
    df=pd.DataFrame(df_dict)
    df.to_csv(save_path,index=0)












