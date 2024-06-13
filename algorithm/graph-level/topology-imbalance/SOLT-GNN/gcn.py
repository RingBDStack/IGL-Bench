from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_add_pool, SAGEConv, global_mean_pool
from torch.nn import Linear
import torch
import torch.nn.functional as F
from util import convert_to_pyg_data


class GCN(torch.nn.Module):
    def __init__(self, num_layers, in_channels, hidden_channels, out_channels, dropout, device):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.lin = Linear(hidden_channels, out_channels)
        self.final_dropout = dropout
        self.device = device

    def forward(self, batch_graph_list):
        data = convert_to_pyg_data(batch_graph_list, self.device)
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.final_dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        h = global_mean_pool(x, batch)  # Global Pooling
        x = self.lin(h)
        return x

    def get_graph_repre(self, batch_graph_list):
        data = convert_to_pyg_data(batch_graph_list, self.device)
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.final_dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        x = global_mean_pool(x, batch)  # Global Pooling
        return x

    def get_patterns(self, batch_graph_list):
        data = convert_to_pyg_data(batch_graph_list, self.device)
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.final_dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

    def predict(self, repre):
        return self.lin(repre)

    def __preprocess_subgraph(self, batch_graph):
        pyg_data_list = []
        idx = []
        elem = []
        start_idx = [0]
        for i, graph in enumerate(batch_graph):
            nodelist = graph.sample_list
            x = graph.node_features[nodelist].clone().detach().to(torch.float)
            edge_index = graph.edge_mat
            # 创建一个布尔掩码，表示哪些边的起点和终点都在 nodelist 中
            nodelist_tensor = torch.tensor(nodelist)
            mask = torch.isin(edge_index[0], nodelist_tensor) & torch.isin(edge_index[1], nodelist_tensor)

            # 根据掩码过滤边
            edge_index = edge_index[:, mask]

            # 重映射边的索引
            node_dict = {old: new for new, old in enumerate(nodelist)}
            edge_index = torch.tensor(
                [[node_dict[i.item()] for i in edge_index[0]], [node_dict[i.item()] for i in edge_index[1]]],
                dtype=torch.long)

            pyg_data_list.append(Data(x=x, edge_index=edge_index))
            elem.extend([1] * len(nodelist))
            idx.extend([[i, j] for j in range(start_idx[i], start_idx[i] + len(nodelist))])
            start_idx.append(start_idx[i] + len(nodelist))

        elem = torch.FloatTensor(elem)
        idx = torch.LongTensor(idx).transpose(0, 1)
        graph_pool = torch.sparse.FloatTensor(idx, elem, torch.Size([len(batch_graph), start_idx[-1]])).to(self.device)

        batched_data = Batch.from_data_list(pyg_data_list)
        return batched_data.to(self.device), graph_pool

    def subgraph_rep(self, batch_graph):
        batched_data, graph_pool = self.__preprocess_subgraph(batch_graph)
        x, edge_index, batch = batched_data.x, batched_data.edge_index, batched_data.batch
        h = x
        for conv in self.convs[:-1]:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, self.final_dropout, training=self.training)
        h = self.convs[-1](h, edge_index)
        pooled_h = torch.spmm(graph_pool, h)

        return pooled_h
