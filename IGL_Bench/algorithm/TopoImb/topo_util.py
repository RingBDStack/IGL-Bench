import os
import math
import numpy as np

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import WLConv
from torch_geometric.loader import DataLoader

from sklearn.cluster import SpectralClustering
from IGL_Bench.algorithm.TopoImb.trainer import GClsTrainer

def clust(features, n_clusters=8):
    clustering = SpectralClustering(
        n_clusters=n_clusters,
        assign_labels='discretize',
        random_state=0,
        affinity='nearest_neighbors'
    ).fit(features)

    return clustering.labels_

class WLGraph_model(torch.nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, dropout, nlayer=2, res=True):
        super().__init__()
        self.args = args
        self.nhid = nhid
        self.res = res

        self.convs = torch.nn.ModuleList()
        for layer in range(nlayer):
            self.convs.append(WLConv())

        self.emb = torch.nn.Embedding(5000, 32)

        self.color_size = -1
        self.graph_map = {}

        self.lin1 = Linear(32, nhid)
        self.lin2 = Linear(nhid, nclass)

        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        if batch is None:  # No batch given
            print('no batch info given')
            batch = x.new(x.size(0)).long().fill_(0)

        x = self.embedding(x, edge_index, edge_weight, batch=batch)
        x = self.lin2(F.leaky_relu(self.lin1(x)))
        return F.log_softmax(x, dim=1)

    def embedding(self, x, edge_index, edge_weight=None, batch=None):
        if x.shape[-1] != 1:
            x = x.argmax(-1)

        for gconv in self.convs:
            x = gconv(x, edge_index)

        out = []
        for b_i in set(batch.cpu().numpy()):
            b_idx = (batch == b_i)
            g_i = x[b_idx]
            idx = hash(tuple(g_i.cpu().numpy().tolist()))
            if idx not in self.graph_map:
                self.graph_map[idx] = len(self.graph_map)
            out.append(self.graph_map[idx])
        g_x = torch.tensor(out, device=x.device)

        if self.color_size == -1:
            self.color_size = len(self.graph_map)

        g_x = self.emb(g_x)
        gx = F.dropout(g_x, self.dropout, training=self.training)
        return gx

    def wl(self, x, edge_index, batch=None):
        if x.shape[-1] != 1:
            x = x.argmax(-1)

        for gconv in self.convs:
            x = gconv(x, edge_index)

        out = []
        for b_i in set(batch.cpu().numpy()):
            b_idx = batch == b_i
            g_i = x[b_idx]
            idx = hash(tuple(g_i.cpu().numpy().tolist()))
            if idx not in self.graph_map:
                self.graph_map[idx] = len(self.graph_map)
            out.append(self.graph_map[idx])
        g_x = torch.tensor(out, device=x.device)
        return g_x

    def graph_wl_dist(self, x, edge_index, batch=None):
        if x.shape[-1] != 1:
            x = x.argmax(-1)

        for gconv in self.convs:
            x = gconv(x, edge_index)
            
        out = self.convs[-1].histogram(x, batch, norm=True)  # (batch_size, num_colors)
        out = out.to(x.device)
        return out


def generate_topo_labels(dataset, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    allloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    steps = math.ceil(len(dataset) / config.batch_size)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    topo_path = os.path.join(current_dir, f"../../../TopoImb_topo_file/{dataset.name}_topo_labels.npy")
    print(f"Checking path: {os.path.abspath(topo_path)}")

    if os.path.exists(topo_path):
        topo_labels_np = np.load(topo_path)
        topo_labels = torch.tensor(topo_labels_np, dtype=torch.long, device=device)
        return topo_labels
    
    wlmodel = WLGraph_model(
        args=config,
        nfeat=dataset.num_features,
        nhid=config.hidden_dim,
        nclass=dataset.num_classes,
        dropout=0,
        nlayer=config.n_layer
    ).to(device)

    WLtrainer = GClsTrainer(config, wlmodel, dataset=dataset)

    for epoch in range(5):
        for batch, data in enumerate(allloader):
            log_info = WLtrainer.train_step(data.to(device), epoch)
            # print(f"[Epoch {epoch}] Train log: {log_info}")

    wlmodel.eval()
    wl_dists = []
    for batch, data in enumerate(allloader):
        graph_wl_tensor = wlmodel.graph_wl_dist(
            data.x.float().to(device),
            data.edge_index.to(device),
            batch=data.batch.to(device)
        ).detach()
        wl_dists.append(graph_wl_tensor.cpu().numpy())

    wl_dists = np.concatenate(wl_dists, axis=0)

    graph_clust = clust(wl_dists, n_clusters=8)
    topo_labels = torch.tensor(graph_clust, device=device)

    # torch.save(topo_labels, topo_path)
    topo_dir = os.path.dirname(topo_path)
    os.makedirs(topo_dir, exist_ok=True)
    
    np.save(topo_path, topo_labels.cpu().numpy())

    return topo_labels
