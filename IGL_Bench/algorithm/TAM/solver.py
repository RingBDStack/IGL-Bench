from IGL_Bench.backbone.gcn import GCN_node_sparse
from IGL_Bench.algorithm.TAM.tam import *
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

class TAM_node_solver:
    def __init__(self, config, dataset, device='cuda'):
        self.config = config
        self.dataset = dataset
        self.device = device

        num_classes = self.dataset.y.numpy().max().item() + 1
        stats = dataset.y[dataset.train_mask]
        n_data = []
        for i in range(num_classes):
            data_num = (stats == i).sum()
            n_data.append(int(data_num.item()))
        self.class_num_list = n_data
        
        self.criterion = torch.nn.CrossEntropyLoss()
        
        self.model = {}
        self.optimizer = {}
        self.initializtion()
        self.aggregator = MeanAggregation()
        
        self.model['default'] = self.model['default'].to(device)  
        self.dataset = self.dataset.to(self.device) 
        
    def initializtion(self):
        num_classes = self.dataset.y.numpy().max().item() + 1
        self.model['default'] = GCN_node_sparse(n_feat=self.dataset.num_features, 
                                          n_hidden=self.config.hidden_dim, 
                                          n_class=num_classes, 
                                          n_layer=self.config.n_layer,dropout=self.config.dropout)
        
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
        least_epoch = getattr(self.config, 'least_epoch', 100)

        best_loss = float('inf')
        patience_counter = 0
        best_val_accuracy = 0

        for epoch in range(1, num_epochs + 1):
            self.model['default'].train()
            self.optimizer['default'].zero_grad()
            
            output = self.model['default'](self.dataset.x, self.dataset.edge_index)
            output = adjust_output(self.config, output, self.dataset.edge_index, self.dataset.y, \
                self.dataset.train_mask, self.aggregator, self.class_num_list, epoch)
            
            loss = self.criterion(output, self.dataset.y[self.dataset.train_mask])
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
        """ Evaluate the model on the validation or test set using the selected metric. """
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