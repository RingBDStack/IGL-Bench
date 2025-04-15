import copy

from IGL_Bench.algorithm.RAWLSGCN.RawlsGCN import RawlsGCNGraph, RawlsGCNGrad
from IGL_Bench.algorithm.RAWLSGCN.utils import matrix2tensor, symmetric_normalize, get_doubly_stochastic, tensor2matrix
from IGL_Bench.backbone.gcn import GCN_node_sparse
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score


class RAWLSGCN_node_solver:
    def __init__(self, config, dataset, device='cuda'):
        self.config = config
        self.dataset = dataset
        self.device = device

        self.model = {}
        self.optimizer = {}
        self.initialization()

        self.model['default'] = self.model['default'].to(device)
        self.dataset = self.dataset.to(self.device)
        self.graph = []

    def initialization(self):
        num_classes = self.dataset.y.numpy().max().item() + 1
        self.process_data()

        if self.config.model == "rawlsgcn_graph":
            self.model['default'] = RawlsGCNGraph(
                nfeat=self.dataset.num_features,
                nhid=self.config.hidden,
                nclass=num_classes,
                dropout=self.config.dropout,
            )
        elif self.config.model == "rawlsgcn_grad":
            self.model['default'] = RawlsGCNGrad(
                nfeat=self.dataset.num_features,
                nhid=self.config.hidden,
                nclass=num_classes,
                dropout=self.config.dropout,
            )

        self.optimizer['default'] = torch.optim.Adam(self.model['default'].parameters(), lr=self.config.lr,
                                                     weight_decay=self.config.weight_decay)

    def process_data(self):
        if self.config.model == "rawlsgcn_grad":
            self.graph = matrix2tensor(
                symmetric_normalize(self.dataset.adj)
            )
        elif self.config.model == "rawlsgcn_graph":
            self.graph = symmetric_normalize(self.dataset.adj)
            self.graph = get_doubly_stochastic(self.graph)

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
        criterion = None

        if self.config.loss == "negative_log_likelihood":
            criterion = torch.nn.NLLLoss()
        elif self.config.loss == "cross_entropy":
            criterion = torch.nn.CrossEntropyLoss()

        if self.config.model == "rawlsgcn_grad":
            self.train_rawls_grad(criterion)
        elif self.config.model == "rawlsgcn_graph":
            self.train_rawls_graph(criterion)

        print("Training Finished!")

    def train_rawls_grad(self, criterion):
        best_model_state_dict = []
        best_val_accuracy = 0.0
        patience_counter = 0

        num_epochs = getattr(self.config, 'epoch', 500)
        patience = getattr(self.config, 'patience', 100)
        least_epoch = getattr(self.config, 'least_epoch', 40)

        for epoch in range(num_epochs):
            self.model['default'].train()
            self.optimizer['default'].zero_grad()

            # training
            pre_act_embs, embs = self.model['default'](self.dataset.x, self.dataset.adj)
            loss_train = criterion(
                embs[-1][self.dataset.train_mask], self.dataset.y[self.dataset.train_mask]
            )
            loss_train.backward()
            self._fix_gradient(pre_act_embs, embs)
            self.optimizer['default'].step()

            print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss_train.item():.4f}")

            # validation
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

    def _fix_gradient(self, pre_act_embs, embs):
        flag = 0  # flag = 0 for weight, flag = 1 for bias
        weights, biases = list(), list()
        # group params
        for name, param in self.model['default'].named_parameters():
            layer, param_type = name.split(".")
            if param_type == "weight":
                if flag == 1:
                    flag = 0
                weights.append(param.data)
            else:
                if flag == 0:
                    flag = 1
                biases.append(param.data)
            flag = 1 - flag

        # fix gradient
        for name, param in self.model['default'].named_parameters():
            layer, param_type = name.split(".")
            idx = self.model['default'].layers_info[layer]
            # idx for embs and pre_act_embs are aligned here because we add a padding in embs (i.e., input features)
            if param_type == "weight":
                normalized_grad = torch.mm(
                    embs[idx].transpose(1, 0),
                    torch.sparse.mm(
                        get_doubly_stochastic(tensor2matrix(self.graph)), pre_act_embs[idx].grad
                    ),
                )
            else:
                normalized_grad = torch.squeeze(
                    torch.mm(
                        torch.ones(1, self.dataset.num_nodes).to(self.device),
                        pre_act_embs[idx].grad,
                    )
                )
            param.grad = normalized_grad

    def train_rawls_graph(self, criterion):
        best_model_state_dict = []
        best_val_accuracy = 0.0
        patience_counter = 0

        num_epochs = getattr(self.config, 'epoch', 500)
        patience = getattr(self.config, 'patience', 100)
        least_epoch = getattr(self.config, 'least_epoch', 40)

        for epoch in range(num_epochs):
            self.model['default'].train()
            self.optimizer['default'].zero_grad()

            # training
            #TODO norm or not?
            output = self.model['default'](self.dataset.x, self.dataset.adj)
            loss_train = criterion(
                output[self.dataset.train_mask], self.dataset.y[self.dataset.train_mask]
            )
            loss_train.backward()
            self.optimizer['default'].step()

            print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss_train.item():.4f}")

            # validation
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

    def eval(self, metric="accuracy"):
        """ Evaluate the model on the validation or test set using the selected metric. """
        self.model['default'].eval()
        all_labels = self.dataset.y[self.dataset.val_mask].cpu().numpy()

        with torch.no_grad():
            out = self.model['default'](self.dataset.x, self.dataset.adj)
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
        self.model['default'].eval()
        all_labels = self.dataset.y[self.dataset.test_mask].cpu().numpy()

        with torch.no_grad():
            out = self.model['default'](self.dataset.x, self.dataset.adj)
            predictions = out[self.dataset.val_mask].argmax(dim=1).cpu().numpy()
            probabilities = torch.nn.functional.softmax(out[self.dataset.test_mask], dim=1).cpu().numpy()

        accuracy = accuracy_score(all_labels, predictions)
        macro_f1 = f1_score(all_labels, predictions, average='macro')
        bacc = balanced_accuracy_score(all_labels, predictions)
        auc_roc = roc_auc_score(all_labels, probabilities, multi_class='ovr', average='macro')

        return accuracy, bacc, macro_f1, auc_roc
