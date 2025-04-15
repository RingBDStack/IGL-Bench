import numpy as np
import torch

def index2adj_bool(edge_index,nnode):

    indx = edge_index.numpy()
    adj = np.zeros((nnode,nnode),dtype = 'bool')
    adj[(indx[0],indx[1])]=1
    new_adj = torch.from_numpy(adj)
    
    return new_adj