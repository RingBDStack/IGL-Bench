import dgl
import torch
import math
import numpy as np


class Graph_Dataset(torch.utils.data.IterableDataset):
    def __init__(self, g, cutoff, node_drop_ratios, sample_size=5, k=3):
        self.g = dgl_graph_copy(g, set_feat=True)
        for item in list(self.g.ndata.keys()):
            if item == 'feat': pass
            else: del self.g.ndata[item]
        self.k = k
        self.sample_size = sample_size
        self.node_drop_ratios = node_drop_ratios
        dst = g.all_edges()[1]
        self.num_neighbors = (torch.unique(dst, return_counts=True)[1] - 1).numpy()
        self.avg_degree = np.mean(self.num_neighbors)
        available_indices = (torch.tensor(self.num_neighbors)>=cutoff).nonzero()[:,0]
        self.available_indices = available_indices[torch.randperm(len(available_indices))]
        self.start = 0
        self.end = len(self.available_indices)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            self.iter_start = self.start
            self.iter_end = self.end
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            self.iter_start = self.start + worker_id * per_worker
            self.iter_end = min(self.iter_start + per_worker, self.end)
        
        return iter(self.get_stream())

    def get_stream(self):
        while True:
            lst = self.available_indices[self.iter_start:self.iter_end][torch.randperm(self.iter_end-self.iter_start)]
            for l in lst:
                yield self.get(l) 
    
    def get(self, idx):

        data = dgl.khop_in_subgraph(self.g, [idx], k=self.k, store_ids=False)
        graph, inverse_index = data[0], data[1]
        neighbors = graph.predecessors(inverse_index)
        neighbors = neighbors[neighbors != inverse_index]
        gdata = [graph, inverse_index, neighbors]
        gdata_total = [gdata[:-1]]
        for i, node_drop_ratio in enumerate(self.node_drop_ratios):
            masked_graphs = []
            inverse_indices_masked = []
            for _ in range(self.sample_size if i != (len(self.node_drop_ratios)-1) else 1):
                if len(gdata[2]) == 0:
                    print(idx)
                    raise Warning('Node with no neighbor is selected. This should not happen.')
                    masked_graph = dgl_graph_copy(gdata[0], set_feat=True)
                    inverse_index_masked = torch.tensor([0])
                else:
                    mask = torch.rand(len(gdata[2])).uniform_()<node_drop_ratio
                    if mask.sum() != 0:
                        temp_g = dgl.remove_nodes(gdata[0], 
                                            gdata[2][mask],
                                            store_ids=True
                                            )
                        temp = temp_g.khop_in_subgraph([(temp_g.ndata[dgl.NID] == gdata[1]).nonzero().squeeze().item()], k=self.k)
                        masked_graph = temp[0]
                        inverse_index_masked = temp[1]
                    elif mask.sum() == len(mask):
                        temp = dgl.khop_in_subgraph(gdata[0], [gdata[1]], 0)
                        masked_graph = temp[0]
                        inverse_index_masked = temp[1]
                    else:
                        temp_g = dgl_graph_copy(gdata[0], set_feat=True)
                        masked_graph = temp_g
                        inverse_index_masked = gdata[1]

                if '_ID' in masked_graph.ndata: del masked_graph.ndata['_ID']
                if '_ID' in masked_graph.edata: del masked_graph.edata['_ID']

                masked_graphs.append(dgl.to_bidirected(masked_graph, copy_ndata=True))
                inverse_indices_masked.append(inverse_index_masked)
            batched_graphs = dgl.batch(masked_graphs)
            inverse_indices_masked = torch.cat(inverse_indices_masked)
            gdata_total.insert(0, [batched_graphs, inverse_indices_masked])
        return gdata_total
    
class Graph_Dataset_splits(torch.utils.data.Dataset):
    def __init__(self, g, splits, k=3):
        self.g = g
        self.k = k
        self.available_indices = splits
        self.len = len(self.available_indices)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data = dgl.khop_in_subgraph(self.g, [self.available_indices[idx]], k=self.k)
        graph, inverse_index = data[0], data[1][0]
        neighbors = graph.predecessors(inverse_index)
        neighbors = neighbors[neighbors != inverse_index]
        return graph, inverse_index, len(neighbors)


class Graph_Collator_infer(object):

    def __call__(self, graphs):

        # unmasked graph processing
        batched_graphs = dgl.batch([g[0] for g in graphs])
        inverse_indices = torch.stack([g[1] for g in graphs]).squeeze()
        batch_num_nodes = batched_graphs.batch_num_nodes()
        offset = torch.cat([torch.tensor([0]), torch.cumsum(batch_num_nodes, dim=0)[:-1]]) + inverse_indices
        return batched_graphs, offset

class Graph_Collator_train(object):

    def __call__(self, gdata):
        diffusion_step = len(gdata[0])
        masked_graphs = []
        inverse_indices = []
        
        for t in range(diffusion_step):
            batch_graphs = dgl.batch([data[t][0] for data in gdata])
            this_inverse_indices = torch.cat([data[t][1] for data in gdata])
            batch_num_nodes = batch_graphs.batch_num_nodes()
            offset = torch.cat([torch.tensor([0]), torch.cumsum(batch_num_nodes, dim=0)[:-1]]) + this_inverse_indices
            inverse_indices.append(offset)
            masked_graphs.append(batch_graphs)
        
        return masked_graphs, inverse_indices
        
def dgl_graph_copy(g, set_feat=False):
    ndata = g.ndata
    edges = g.edges()
    new_g = dgl.graph(edges)
    if set_feat:
        for k, v in ndata.items():
            new_g.ndata[k] = v
    return new_g