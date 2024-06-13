#Calculate Hyperbolic Embedding
import argparse
import torch
import numpy as np
from models.Poincare import PoincareModel
import dataloader as dl
from torch_geometric.utils import degree, to_networkx
from GraphRicciCurvature.OllivierRicci import OllivierRicci

parser = argparse.ArgumentParser(description='Calculate Hyperbolic Embedding')
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--manifolds', type=str, default='ricci', help="ricci, poincare")
parser.add_argument("--dataset", '-d', type=str, default="Cora", help="all,Cora,Citeseer,Photo,Actor,chameleon,Squirrel")
parser.add_argument("--split", '-s', type=str, default=0, help="Random split train-set")

args = parser.parse_args()
print(args)

dataset,data,_,_,_ = dl.select_dataset(args.dataset, args.split,1)


if args.manifolds=='ricci':
    G = to_networkx(data)
    orc = OllivierRicci(G, alpha=0.5, verbose="TRACE")
    orc.compute_ricci_curvature()
    G_orc = orc.G.copy()  # save an intermediate result

    curvature="ricciCurvature"
    ricci_results = {}
    ricci = {}
    for i,(n1,n2) in enumerate(list(G_orc.edges()),0):
        #ricci_results[i] = G_orc[n1][n2][curvature]
        ricci[i] = [int(n1),int(n2),G_orc[n1][n2][curvature]]

    weights = [ricci[i] for i in ricci.keys()]
    np.savetxt('hyperemb/' + args.dataset + '.edge_list',weights,fmt="%d %d %.16f")

else:
    degrees = np.array(degree(data.edge_index[0],num_nodes=data.num_nodes)+degree(data.edge_index[1],num_nodes=data.num_nodes))
    edges_list = list(data.edge_index.t().numpy())
    labels = dict(enumerate(data.y.numpy()+1, 0))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dim = 2
    model = PoincareModel(edges_list,node_weights=degrees*0.2,node_labels=labels, n_components=dim,eta=0.01,n_negative=10, name="hierarchy", device=device)
    model.init_embeddings()
    model.train(args.epochs)
    weights = model.embeddings
    keys = np.array([item for item in model.emb_dict.keys()])
    values = np.array([item for item in model.emb_dict.values()])
    np.save('hyperemb/' + args.dataset + '_keys.npy', keys)
    np.save('hyperemb/' + args.dataset + '_values.npy', values)
