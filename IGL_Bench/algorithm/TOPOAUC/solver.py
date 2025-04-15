from IGL_Bench.backbone.gcn import GCN_node_sparse
from IGL_Bench.algorithm.TOPOAUC.myloss import ELossFN
from IGL_Bench.algorithm.TOPOAUC.cal import compute_ppr_and_gpr
from IGL_Bench.algorithm.TOPOAUC.util import *
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

class TOPOAUC_node_solver:
    def __init__(self, config, dataset, device='cuda'):
        self.config = config
        self.dataset = dataset
        self.device = device
        
        self.model = {}
        self.optimizer = {}
        self.ppr, self.gpr = compute_ppr_and_gpr(self.dataset, self.config.pagerank_prob)
        self.initializtion()
        
        self.model['default'] = self.model['default'].to(device)  
        self.my_loss = self.my_loss.to(device)
        self.dataset = self.dataset.to(device) 
        
    def initializtion(self):
        num_classes = self.dataset.y.numpy().max().item() + 1
        self.model['default'] = GCN_node_sparse(n_feat=self.dataset.num_features, 
                                          n_hidden=self.config.hidden_dim, 
                                          n_class=num_classes, 
                                          n_layer=self.config.n_layer,dropout=self.config.dropout)
        
        adj_bool=index2adj_bool(self.dataset.edge_index,self.dataset.num_nodes)
        
        self.my_loss=ELossFN(num_classes,self.dataset.num_nodes,adj_bool,self.ppr,self.gpr,self.dataset.train_mask,
                        self.device,weight_sub_dim=self.config.weight_sub_dim,weight_inter_dim=self.config.weight_inter_dim,
                        weight_global_dim=self.config.weight_global_dim,beta= self.config.beta,gamma=self.config.gamma,
                        loss_type=self.config.loss)
        
        self.optimizer['default'] = torch.optim.Adam(self.model['default'].parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        
    def reset_parameters(self):
        for model_name, model in self.model.items():
            if hasattr(model, 'reset_parameters'):
                model.reset_parameters()
            else:
                for layer in model.modules():
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()
                        
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
        patience = getattr(self.config, 'patience', 10)
        least_epoch = getattr(self.config, 'least_epoch', 40)
        
        criterion = torch.nn.CrossEntropyLoss()

        best_loss = float('inf')
        patience_counter = 0
        best_val_accuracy = 0

        for epoch in range(1, num_epochs + 1):
            self.model['default'].train()
            self.optimizer['default'].zero_grad()

            out = self.model['default'](self.dataset.x, self.dataset.edge_index)
            logits = F.softmax(out, dim=-1)
            loss = self.my_loss(logits, self.dataset.y,self.dataset.train_mask)
            loss = torch.mean(loss)
            loss.backward()
            self.optimizer['default'].step()

            print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}")

            val_accuracy = self.eval(metric="accuracy")

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience and epoch > least_epoch:
                print(f"Early stopping at epoch {epoch+1}.")
                break

        print("Training Finished!")
        
    def eval(self, metric="accuracy"):
        self.model['default'].eval()
        all_labels = self.dataset.y[self.dataset.val_mask].cpu().numpy()

        with torch.no_grad():
            out = self.model['default'](self.dataset.x, self.dataset.edge_index)
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
            out = self.model['default'](self.dataset.x, self.dataset.edge_index)
            predictions = out[self.dataset.test_mask].argmax(dim=1).cpu().numpy()
            probabilities = torch.nn.functional.softmax(out[self.dataset.test_mask], dim=1).cpu().numpy()

        accuracy = accuracy_score(all_labels, predictions)
        macro_f1 = f1_score(all_labels, predictions, average='macro')
        bacc = balanced_accuracy_score(all_labels, predictions)
        auc_roc = roc_auc_score(all_labels, probabilities, multi_class='ovr', average='macro')
        
        return accuracy, bacc, macro_f1, auc_roc