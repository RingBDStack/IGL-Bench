import copy

import scipy as sp
from torch import nn

from IGL_Bench.algorithm.TAILGNN.TailGNN import TailGCN_SP, Discriminator
from IGL_Bench.algorithm.TAILGNN.util import link_dropout, normalize, convert_sparse_tensor
from IGL_Bench.backbone.gcn import GCN_node_sparse
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score


class TAILGNN_node_solver:
    def __init__(self, config, dataset, device='cuda'):
        self.config = config
        self.dataset = dataset
        self.device = device

        self.model = {}
        self.optimizer = {}
        self.initialization()

        self.model['default'] = self.model['default'].to(device)
        self.dataset = self.dataset.to(self.device)

        self.adj, self.tail_adj, self.adj_self, self.tail_adj_self, self.norm_adj_self, self.norm_tail_adj_self = [], [], [], [], [], []
        self.h_labels, self.t_labels = [], []

    def initialization(self):
        num_classes = self.dataset.y.numpy().max().item() + 1
        self.process_data()

        self.model['default'] = TailGCN_SP(nfeat=self.dataset.num_features,
                                            nclass=num_classes,
                                            params=self.config,
                                            device=self.device)

        self.optimizer['default'] = torch.optim.Adam(self.model['default'].parameters(), lr=self.config.lr,
                                                     weight_decay=self.config.weight_decay)

        self.model['disc'] = Discriminator(num_classes)

        self.optimizer['disc'] = torch.optim.Adam(self.model['disc'].parameters(), lr=self.config.lr,
                                                     weight_decay=self.config.weight_decay)

    def process_data(self):
        x, edge_index = self.dataset.x, self.dataset.edge_index
        adj, adj_norm = self.dataset.adj, self.dataset.adj_norm
        adj = adj - sp.eye(adj.shape[0])
        train_index = self.dataset.train_index
        tail_adj = link_dropout(adj, train_index)

        adj = normalize(adj)
        tail_adj = normalize(tail_adj)
        self.adj = convert_sparse_tensor(adj)  # torch.FloatTensor(adj.todense())
        self.tail_adj = convert_sparse_tensor(tail_adj)  # torch.FloatTensor(tail_adj.todense())

        self.adj_self, norm_adj_self = self.dataset.adj, self.dataset.adj_norm
        self.tail_adj_self = tail_adj + sp.sparse.eye(adj.shape[0])
        self.norm_adj_self = torch.unsqueeze(torch.sparse.sum(self.adj_self, dim=1).to_dense(), 1)
        self.norm_tail_adj_self = torch.unsqueeze(torch.sparse.sum(self.tail_adj_self, dim=1).to_dense(), 1)

        self.h_labels = torch.full((len(self.dataset.train_mask), 1), 1.0, device=self.device)
        self.t_labels = torch.full((len(self.dataset.train_mask), 1), 0.0, device=self.device)

    def reset_parameters(self):
        """Reset model parameters and reinitialize optimizer."""
        # Reset model parameters
        for model_name, model in self.model.items():
            if hasattr(model, 'reset_parameters'):
                model.reset_parameters()
            else:
                for layer in model.modules():
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()

        # Reinitialize optimizer
        self.optimizer = {}
        for model_name, model in self.model.items():
            self.optimizer[model_name] = torch.optim.Adam(
                model.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.weight_decay
            )

    def train(self):
        self.reset_parameters()
        num_epochs = getattr(self.config, 'epoch', 500)
        patience = getattr(self.config, 'patience', 100)
        least_epoch = getattr(self.config, 'least_epoch', 40)

        criterion = nn.BCELoss()

        best_loss = float('inf')
        patience_counter = 0
        best_val_accuracy = 0
        best_model_state_dict = []

        for epoch in range(1, num_epochs + 1):
            self.model['default'].train()
            self.optimizer['default'].zero_grad()

            L_d = self.train_disc(criterion)
            L_all, L_cls, L_d = self.train_embed(criterion)

            print(f"Epoch [{epoch}/{num_epochs}], Loss: {L_all.item():.4f}")

            val_accuracy = self.eval(metric="accuracy")

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0
                best_model_state_dict = copy.deepcopy(self.model['default'].state_dict())
            else:
                patience_counter += 1

            if patience_counter >= patience and epoch > least_epoch:
                print(f"Early stopping at epoch {epoch + 1}.")
                break

        self.model['default'].load_state_dict(best_model_state_dict)
        print("Training Finished!")

    def train_disc(self, criterion):
        self.model['disc'].train()
        self.optimizer['disc'].zero_grad()

        embed_h, norm1, norm2 = self.model['default'](self.dataset.x, self.adj, True, self.adj_self, self.norm_adj_self)
        embed_t, _, _ = self.model['default'](self.dataset.x, self.tail_adj, False, self.tail_adj_self, self.norm_tail_adj_self)

        prob_h = self.model['disc'](embed_h)
        prob_t = self.model['disc'](embed_t)

        # loss
        # L_cls = F.nll_loss(F.softmax(embed_h[idx_train], dim=1), labels[idx_train]) + F.nll_loss(F.softmax(embed_t[idx_train], dim=1), labels[idx_train])
        errorD = criterion(prob_h[self.dataset.train_mask], self.h_labels)
        errorG = criterion(prob_t[self.dataset.train_mask], self.t_labels)
        L_d = (errorD + errorG) / 2

        L_d.backward()
        self.optimizer['disc'].step()
        return L_d

    def train_embed(self, criterion):
        self.model['default'].train()
        self.optimizer['default'].zero_grad()

        embed_h, norm1, norm2 = self.model['default'](self.dataset.x, self.adj, True, self.adj_self, self.norm_adj_self)

        embed_t, _, _ = self.model['default'](self.dataset.x, self.tail_adj, False, self.tail_adj_self, self.norm_tail_adj_self)

        # loss
        L_cls_h = F.nll_loss(F.log_softmax(embed_h[self.dataset.train_mask], dim=1), self.dataset.y[self.dataset.train_mask])
        L_cls_t = F.nll_loss(F.log_softmax(embed_t[self.dataset.train_mask], dim=1), self.dataset.y[self.dataset.train_mask])
        L_cls = (L_cls_h + L_cls_t) / 2

        prob_h = self.model['disc'](embed_h)
        prob_t = self.model['disc'](embed_t)

        errorD = criterion(prob_h[self.dataset.train_mask], self.h_labels)
        errorG = criterion(prob_t[self.dataset.train_mask], self.t_labels)
        L_d = errorG

        norm = torch.mean(norm1[self.dataset.train_mask]) + torch.mean(norm2[self.dataset.train_mask])

        L_all = L_cls - (self.config.eta * L_d) + self.config.mu * norm
        # L_all = L_cls + (lambda_d * L_g) + mu * norm

        L_all.backward()
        self.optimizer['default'].step()

        return L_all, L_cls, L_d

    def eval(self, metric="accuracy"):
        """ Evaluate the model on the validation or test set using the selected metric. """
        self.model['default'].eval()
        all_labels = self.dataset.y[self.dataset.val_mask].cpu().numpy()

        with torch.no_grad():
            out, _, _ = self.model['default'](self.dataset.x, self.adj, False, self.adj_self, self.norm_adj_self)
            predictions = out[self.dataset.val_mask].argmax(dim=1).cpu().numpy()

        if metric == "accuracy":
            return accuracy_score(all_labels, predictions)
        elif metric == "bacc":
            return balanced_accuracy_score(all_labels, predictions)
        elif metric == "macro_f1":
            return f1_score(all_labels, predictions, average='macro')
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def test(self):
        all_labels = self.dataset.y[self.dataset.test_mask].cpu().numpy()
        self.model['default'].eval()
        with torch.no_grad():
            out, _, _ = self.model['default'](self.dataset.x, self.adj, False, self.adj_self, self.norm_adj_self)
            predictions = out[self.dataset.test_mask].argmax(dim=1).cpu().numpy()
            probabilities = torch.nn.functional.softmax(out[self.dataset.test_mask], dim=1).cpu().numpy()

        accuracy = accuracy_score(all_labels, predictions)
        bacc = balanced_accuracy_score(all_labels, predictions)
        macro_f1 = f1_score(all_labels, predictions, average='macro')
        auc_roc = auc_roc = roc_auc_score(all_labels, probabilities, multi_class='ovr', average='macro')

        return accuracy, bacc, macro_f1, auc_roc
