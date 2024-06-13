import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.graph_learner import GraphLearner
from ..layers.backbones import GCN, GAT, APPNP, GraphSAGE, ChebNet, SGC
from ..utils.generic_utils import to_cuda
from ..utils.constants import VERY_SMALL_NUMBER


class GraphClf(nn.Module):
    def __init__(self, config):
        super(GraphClf, self).__init__()
        self.config = config
        self.graph_learn = config['graph_learn']
        self.gnn = config['gnn']
        self.device = config['device']
        n_feat = config['num_feat']
        n_class = config['num_class']
        hidden_size = config['hidden_size']
        self.dropout = config['dropout']
        self.graph_skip_conn = config['graph_skip_conn']
        self.graph_include_self = config.get('graph_include_self', True)

        if self.gnn == 'gcn':
            self.encoder = GCN(in_feats=n_feat,
                               n_hidden=hidden_size,
                               n_classes=n_class,
                               graph_hops=2,
                               dropout=self.dropout,
                               batch_norm=config.get('batch_norm', False))

        elif self.gnn == 'gat':
            self.encoder = GAT(in_feats=n_feat,
                               n_hidden=hidden_size,
                               n_classes=n_class,
                               n_layers=4,
                               num_heads=8,
                               activation=F.relu,
                               feat_drop=self.dropout,
                               negative_slope=0.2)

        elif self.gnn == 'graphsage':
            self.encoder = GraphSAGE(in_feats=n_feat,
                                     n_hidden=hidden_size,
                                     n_classes=n_class,
                                     n_layers=3,
                                     activation=F.relu,
                                     dropout=self.dropout,
                                     aggregator_type=config.get('graphsage_agg_type', 'gcn'))

        elif self.gnn == 'appnp':
            self.encoder = APPNP(in_feats=n_feat,
                                 n_hidden=hidden_size,
                                 n_layers=3,
                                 n_classes=n_class,
                                 activation=F.relu,
                                 feat_drop=self.dropout,
                                 edge_drop=0.,
                                 alpha=0,
                                 k=3)

        elif self.gnn == 'chebnet':
            self.encoder = ChebNet(in_feats=n_feat,
                                   n_classes=n_class,
                                   n_hidden=hidden_size,
                                   n_layers=3,
                                   k=3,
                                   bias=True)

        elif self.gnn == 'sgc':
            self.encoder = SGC(in_feats=n_feat,
                               n_hidden=hidden_size,
                               n_classes=n_class,
                               k=3,
                               n_layers=3,
                               activation=F.relu,
                               dropout=self.dropout)

        else:
            raise RuntimeError('Unknown GNN: {}'.format(self.gnn))


        if self.graph_learn:
            graph_learn_fun = GraphLearner
            self.graph_learner = graph_learn_fun(n_feat,
                                                 config['graph_learn_hidden_size'],
                                                 n_nodes=config['num_nodes'],
                                                 n_class=config['num_class'],
                                                 n_anchors=config['num_anchors'],
                                                 topk=config['graph_learn_topk'],
                                                 epsilon=config['graph_learn_epsilon'],
                                                 n_pers=config['graph_learn_num_pers'],
                                                 device=self.device)

            self.graph_learner2 = graph_learn_fun(hidden_size,
                                                  config.get('graph_learn_hidden_size2', config['graph_learn_hidden_size']),
                                                  n_nodes= config['num_nodes'],
                                                  n_class=config['num_class'],
                                                  n_anchors=config['num_anchors'],
                                                  topk=config.get('graph_learn_topk2', config['graph_learn_topk']),
                                                  epsilon=config.get('graph_learn_epsilon2', config['graph_learn_epsilon']),
                                                  n_pers=config['graph_learn_num_pers'],
                                                  device=self.device)

        else:
            self.graph_learner = None
            self.graph_learner2 = None


    def learn_graph(self, graph_learner, node_features, position_encoding, gpr_rank, position_flag, graph_skip_conn=None, graph_include_self=False, init_adj=None):
        if self.graph_learn:
            raw_adj = graph_learner(node_features, position_encoding, gpr_rank, position_flag)
            #assert raw_adj.min().item() >= 0
            adj = raw_adj / torch.clamp(torch.sum(raw_adj, dim=-1, keepdim=True), min=VERY_SMALL_NUMBER)

            if graph_skip_conn in (0, None):
                if graph_include_self:
                    adj = adj + to_cuda(torch.eye(adj.size(0)), self.device)
            else:
                try:
                    adj.mul_(1 - graph_skip_conn)
                    adj.add_(init_adj.mul(graph_skip_conn))
                except RuntimeError as e:
                    init_adj_cpu = init_adj.to('cpu')
                    adj_cpu = adj.to('cpu')
                    adj_cpu.mul_(1 - graph_skip_conn)
                    adj_cpu.add_(init_adj_cpu.mul(graph_skip_conn))
                    adj = adj_cpu.to('cuda')
            return raw_adj, adj

        else:
            raw_adj = None
            adj = init_adj
            return raw_adj, adj


    def forward(self, node_features, init_adj=None):
        node_features = F.dropout(node_features, self.config.get('feat_adj_dropout', 0), training=self.training)
        raw_adj, adj = self.learn_graph(self.graph_learner, node_features, self.graph_skip_conn, init_adj=init_adj)
        adj = F.dropout(adj, self.config.get('feat_adj_dropout', 0), training=self.training)
        node_vec = self.encoder(node_features, adj)
        output = F.log_softmax(node_vec, dim=-1)
        return output, adj