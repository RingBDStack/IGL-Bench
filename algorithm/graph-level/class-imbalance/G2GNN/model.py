import torch
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool, GCNConv,SAGEConv, global_max_pool
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
from torch.nn import Linear
import torch.nn.functional as F
from torch_scatter import segment_csr



class MLP_Classifier(torch.nn.Module):
    def __init__(self, args):
        super(MLP_Classifier, self).__init__()
        self.lin1 = Linear(args.n_hidden, args.n_hidden)
        self.lin2 = Linear(args.n_hidden, args.n_class)

    def forward(self, x):
        x = self.lin1(x).relu()
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.lin2(x)

        return F.log_softmax(x, dim=1)

class GIN(torch.nn.Module):
    def __init__(self, args):
        super(GIN, self).__init__()

        self.conv1 = GINConv(
            Sequential(Linear(args.n_feat, args.n_hidden), BatchNorm1d(args.n_hidden), ReLU(),
                       Linear(args.n_hidden, args.n_hidden), ReLU()))

        self.conv2 = GINConv(
            Sequential(Linear(args.n_hidden, args.n_hidden), BatchNorm1d(args.n_hidden), ReLU(),
                       Linear(args.n_hidden, args.n_hidden), ReLU()))

        self.conv3 = GINConv(
            Sequential(Linear(args.n_hidden, args.n_hidden), BatchNorm1d(args.n_hidden), ReLU(),
                       Linear(args.n_hidden, args.n_hidden), ReLU()))

        self.conv4 = GINConv(
            Sequential(Linear(args.n_hidden, args.n_hidden), BatchNorm1d(args.n_hidden), ReLU(),
                       Linear(args.n_hidden, args.n_hidden), ReLU()))

        self.conv5 = GINConv(
            Sequential(Linear(args.n_hidden, args.n_hidden), BatchNorm1d(args.n_hidden), ReLU(),
                       Linear(args.n_hidden, args.n_hidden), ReLU()))


    def forward(self, x, adj_t, batch):
        x = self.conv1(x, adj_t)
        x = self.conv2(x, adj_t)
        x = self.conv3(x, adj_t)
        x = self.conv4(x, adj_t)
        x = self.conv5(x, adj_t)

        x = segment_csr(x, batch, reduce="mean")

        return x
    
class GCN(torch.nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()

        # Define the first GCN layer with input features to hidden features
        self.conv1 = GCNConv(args.n_feat, args.n_hidden)
        
        # Define other GCN layers with hidden features to hidden features
        self.conv2 = GCNConv(args.n_hidden, args.n_hidden)
        self.conv3 = GCNConv(args.n_hidden, args.n_hidden)
        self.conv4 = GCNConv(args.n_hidden, args.n_hidden)
        self.conv5 = GCNConv(args.n_hidden, args.n_hidden)
        
        # Single BatchNorm1d layer for all GCN layers
        self.bn = BatchNorm1d(args.n_hidden)
        self.relu = ReLU()

    def forward(self, x, edge_index, batch):

        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.conv3(x, edge_index)
        x = self.relu(x)
        # x = self.conv4(x, edge_index)
        # x = self.relu(x)
        # x = self.conv5(x, edge_index)
        # x = self.relu(self.bn(x))
        x = segment_csr(x, batch, reduce="mean")
        # x = global_mean_pool(x,batch)

        return x

class SAGE(torch.nn.Module):
    def __init__(self, args):
        super(SAGE, self).__init__()

        # Define the first GraphSAGE layer with input features to hidden features
        self.conv1 = SAGEConv(args.n_feat, args.n_hidden)
        
        # Define other GraphSAGE layers with hidden features to hidden features
        self.conv2 = SAGEConv(args.n_hidden, args.n_hidden)
        self.conv3 = SAGEConv(args.n_hidden, args.n_hidden)
        self.conv4 = SAGEConv(args.n_hidden, args.n_hidden)
        self.conv5 = SAGEConv(args.n_hidden, args.n_hidden)
        
        # Single BatchNorm1d layer for all GraphSAGE layers
        self.bn = BatchNorm1d(args.n_hidden)
        self.relu = ReLU()

    def forward(self, x, edge_index, batch):
        # Apply GraphSAGE layers with ReLU and BatchNorm after each
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        x = self.conv4(x, edge_index)
        x = self.conv5(x, edge_index)
        x = self.relu(self.bn(x))
        x = segment_csr(x, batch, reduce="sum")

        return x