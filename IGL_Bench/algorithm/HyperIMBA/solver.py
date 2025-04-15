from IGL_Bench.algorithm.HyperIMBA.cal import compute_ricci_and_poincare
import IGL_Bench.algorithm.HyperIMBA.GcnHyper as GcnHyper
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

class HyperIMBA_node_solver:
    def __init__(self, config, dataset, device='cuda'):
        self.config = config
        self.dataset = dataset
        self.device = device
        
        compute_ricci_and_poincare(self.dataset)
        
        self.model = {}
        self.optimizer = {}
        self.initialization()
        self.dataset = self.dataset.to(self.device) 
        
    def initialization(self):
        if self.config.backbone == 'GCN':
            self.model['default'], self.dataset = GcnHyper.set_model(self.dataset, self.config)
            
        self.optimizer['default'] = torch.optim.Adam(self.model['default'].parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        
    def reset_parameters(self):
        self.initialization()
        
    def train(self):
        self.reset_parameters()
        num_epochs = getattr(self.config, 'epoch', 500)
        patience = getattr(self.config, 'patience', 50)
        least_epoch = getattr(self.config, 'least_epoch', 40)
        best_val_accuracy = 0
        
        for epoch in range(1, num_epochs + 1):
            self.model['default'].train()
            self.optimizer['default'].zero_grad()

            output = self.model['default'](self.dataset, self.config.loss_hp)
            loss = F.cross_entropy(output[self.dataset.train_mask], self.dataset.y[self.dataset.train_mask])
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
            out = self.model['default'](self.dataset, self.config.loss_hp)
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
            out = self.model['default'](self.dataset, self.config.loss_hp)
            predictions = out[self.dataset.test_mask].argmax(dim=1).cpu().numpy()
            probabilities = torch.nn.functional.softmax(out[self.dataset.test_mask], dim=1).cpu().numpy()

        accuracy = accuracy_score(all_labels, predictions)
        macro_f1 = f1_score(all_labels, predictions, average='macro')
        bacc = balanced_accuracy_score(all_labels, predictions)
        auc_roc = roc_auc_score(all_labels, probabilities, multi_class='ovr', average='macro')
        
        return accuracy, bacc, macro_f1, auc_roc            