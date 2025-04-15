from IGL_Bench.backbone.gin import GIN_graph
import torch
from torch_geometric.loader import DataLoader as GeoDataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, balanced_accuracy_score

def compute_metrics(y_true, y_pred, y_prob=None):
    accuracy = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    if y_prob is not None:
        auc_roc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    else:
        auc_roc = None
    
    return accuracy, bacc, macro_f1, auc_roc

class GIN_graph_solver:
    def __init__(self, config, dataset, device='cuda'):
        self.config = config
        self.dataset = dataset
        self.device = device
        
        self.model = {}
        self.optimizer = {}
        self.initializtion()
        
        for data in self.dataset:
            data.to(device)
        self.model['default'] = self.model['default'].to(device)
        
        self.prepare_data_loaders(device)

        
    def initializtion(self):
        self.model['default'] = GIN_graph(n_feat=self.dataset.num_features, 
                                          n_hidden=self.config.hidden_dim, 
                                          n_class=self.dataset.num_classes, 
                                          n_layer=self.config.n_layer,dropout=self.config.dropout)
        
        self.optimizer['default'] = torch.optim.Adam(self.model['default'].parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)

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
        min_delta = getattr(self.config, 'min_delta', 0.0)

        model = self.model['default']
        optimizer = self.optimizer['default']
        criterion = torch.nn.CrossEntropyLoss()

        best_loss = float('inf')
        patience_counter = 0
        best_val_accuracy = 0

        for epoch in range(1, num_epochs + 1):
            model.train()
            total_loss = 0

            for batch, data in enumerate(self.train_loader): 
                if data.y.dim() > 1:
                    data.y = data.y.view(-1) 
                data = data.to(self.device)
                optimizer.zero_grad()
                out = model(data.x, data.adj_t, data.batch)
                loss = criterion(out, data.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {avg_loss:.4f}")

            val_accuracy = self.eval(self.val_loader, metric="accuracy")

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1}.")
                break

        print("Training Finished!")
        
    def prepare_data_loaders(self, device='cuda'):
        batch_size = getattr(self.config, 'batch_size', 128)

        train_data = self.dataset[self.dataset.train_mask]
        val_data = self.dataset[self.dataset.val_mask]
        test_data = self.dataset[self.dataset.test_mask]

        self.train_loader = GeoDataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.val_loader = GeoDataLoader(val_data, batch_size=batch_size, shuffle=False)
        self.test_loader = GeoDataLoader(test_data, batch_size=batch_size, shuffle=False)
        
    def eval(self, loader, metric="accuracy"):
        """ Evaluate the model on the validation or test set using the selected metric. """
        self.model['default'].eval()
        all_labels = []
        all_preds = []
        all_probs = []  # To store probabilities for AUC-ROC

        with torch.no_grad():
            for batch, data in enumerate(loader):
                data = data.to(self.device)
                x, y = data.x, data.y  # Assuming data contains features (x) and labels (y)
                output = self.model['default'](data.x, data.adj_t, data.batch)

                _, predicted = torch.max(output, dim=1)
                all_labels.append(y.cpu().numpy())
                all_preds.append(predicted.cpu().numpy())
                all_probs.append(torch.nn.Softmax(dim=1)(output).cpu().numpy())  # Get probabilities

        # Convert lists to numpy arrays
        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_preds)
        all_probs = np.concatenate(all_probs)

        # Compute selected metric
        if metric == "accuracy":
            return accuracy_score(all_labels, all_preds)
        elif metric == "bacc":
            return balanced_accuracy_score(all_labels, all_preds)
        elif metric == "macro_f1":
            return f1_score(all_labels, all_preds, average='macro')
        elif metric == "auc_roc":
            return roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
    def test(self):
        """Evaluate the model on the test set and return metrics."""
        self.model['default'].eval()
        all_labels = []
        all_preds = []
        all_probs = []  # To store probabilities for AUC-ROC

        with torch.no_grad():
            for batch, data in enumerate(self.test_loader):
                if data.y.dim() > 1:
                    data.y = data.y.view(-1) 
                data = data.to(self.device)
                output = self.model['default'](data.x, data.adj_t, data.batch)

                _, predicted = torch.max(output, dim=1)
                all_labels.append(data.y.cpu().numpy())
                all_preds.append(predicted.cpu().numpy())
                all_probs.append(torch.nn.Softmax(dim=1)(output).cpu().numpy())  # Get probabilities

        # Convert lists to numpy arrays
        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_preds)
        all_probs = np.concatenate(all_probs)

        # Compute metrics
        acc = accuracy_score(all_labels, all_preds)
        bacc = balanced_accuracy_score(all_labels, all_preds)
        mf1 = f1_score(all_labels, all_preds, average='macro')

        # Compute ROC-AUC
        if len(np.unique(all_labels)) == 2:  # Binary classification
            roc = roc_auc_score(all_labels, all_probs[:, 1])  # Use probabilities of the positive class
        else:  # Multi-class classification
            roc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')

        return acc, bacc, mf1, roc
