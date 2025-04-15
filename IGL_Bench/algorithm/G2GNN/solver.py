from IGL_Bench.backbone.gin import GIN_graph
from IGL_Bench.backbone.gcn import GCN_graph
import torch
# from torch_geometric.loader import DataLoader 
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, balanced_accuracy_score
from IGL_Bench.algorithm.G2GNN.kernel import get_kernel_knn
from IGL_Bench.algorithm.G2GNN.dataloader import Dataset_knn_aug
from IGL_Bench.algorithm.G2GNN.aug import upsample
import torch.nn.functional as F

class G2GNN_graph_solver:
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
        self.config.kernel_idx, self.config.knn_edge_index = get_kernel_knn(self.dataset.name, self.config.kernel_type, 
                                                                           self.config.knn_nei_num,self.dataset)
       
        if self.config.backbone =='GIN':
            gnn_model = GIN_graph(n_feat=self.dataset.num_features, n_hidden=self.config.hidden_dim, 
                                  n_class=self.dataset.num_classes, n_layer=self.config.n_layer,dropout=self.config.dropout)
        elif self.config.backbone =='GCN':
            gnn_model = GCN_graph(n_feat=self.dataset.num_features, n_hidden=self.config.hidden_dim, 
                                  n_class=self.dataset.num_classes, n_layer=self.config.n_layer,dropout=self.config.dropout)
        else:
            raise ValueError('Backbone not supported')

        self.model['default'] = gnn_model.to(self.device) 

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

    def prepare_data_loaders(self, device='cuda'):
        batch_size = getattr(self.config, 'batch_size', 128)
        
        train_index = torch.tensor(self.dataset.train_index, dtype=torch.long)
        val_index = torch.tensor(self.dataset.val_index, dtype=torch.long)
        test_index = torch.tensor(self.dataset.test_index, dtype=torch.long)

        train_data = self.dataset[train_index]
        val_data = self.dataset[val_index]
        test_data = self.dataset[test_index]        
        
        train_data = upsample(train_data)
        val_data = upsample(val_data)    
            
        train_dataset = Dataset_knn_aug(train_data, self.dataset, self.config)
        val_dataset = Dataset_knn_aug(val_data, self.dataset, self.config)
        test_dataset = Dataset_knn_aug(test_data, self.dataset, self.config)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=train_dataset.collate_batch
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=val_dataset.collate_batch
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=test_dataset.collate_batch
        )
        
    def train(self):
        self.reset_parameters()
        num_epochs = getattr(self.config, 'epoch', 500)
        patience = getattr(self.config, 'patience', 10)
        min_delta = getattr(self.config, 'min_delta', 0.0)

        model = self.model['default']
        optimizer = self.optimizer['default']
        criterion = torch.nn.CrossEntropyLoss()

        best_val_accuracy = 0
        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            model.train()

            total_loss = 0
            for batch_idx, batch in enumerate(self.train_loader):
                batch = batch_to_gpu(batch, self.device)
                data, train_idx = batch['data'], batch['train_idx']
                
                knn_adj_t, aug_adj_ts, aug_xs = batch['knn_adj_t'], batch['aug_adj_ts'], batch['aug_xs']
                
                logit_aug_props = []
                for aug_idx in range(self.config.aug_num):
                    H = model.encode(aug_xs[aug_idx], aug_adj_ts[aug_idx], data.batch)
                    H_knn = H
                    for k in range(self.config.knn_layer):
                        H_knn = torch.sparse.mm(knn_adj_t, H_knn)
                    
                    logits = model.cls(H_knn)
                    logits = F.log_softmax(logits, dim=-1)
                    logit_aug_props.append(logits[train_idx])
                
                loss = 0
                for i in range(self.config.aug_num):
                    loss += F.nll_loss(logit_aug_props[i], data.y[train_idx].to(torch.int64))
                loss = loss / self.config.aug_num

                loss = loss + consis_loss(logit_aug_props, temp=self.config.temp)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {avg_loss:.4f}")

            val_accuracy = self.eval(self.val_loader, metric="accuracy")
            print(f"Validation Accuracy: {val_accuracy:.4f}")

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}.")
                break

        print("Training Finished!")
        
    def eval(self, loader, metric="accuracy"):
        model = self.model['default']
        model.eval()

        all_preds = []
        all_labels = []
        all_probs = []  

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                batch = batch_to_gpu(batch, self.device)
                data = batch['data']
                encoder_out = model.encode(data.x, data.adj_t, data.batch)
                logits = model.cls(encoder_out)

                preds = logits.argmax(dim=-1)
                all_preds.append(preds.cpu().numpy())

                all_labels.append(data.y.cpu().numpy())

                probs = F.softmax(logits, dim=-1)
                all_probs.append(probs.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        all_probs = np.concatenate(all_probs)

        if metric == "accuracy":
            score = accuracy_score(all_labels, all_preds)
        elif metric == "bacc":
            score = balanced_accuracy_score(all_labels, all_preds)
        elif metric == "mf1":
            score = f1_score(all_labels, all_preds, average="macro")
        elif metric == "auc-roc":
            if self.dataset.num_classes == 2:
                score = roc_auc_score(all_labels, all_probs[:, 1])
            else:
                score = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        else:
            raise ValueError(f"Unsupported metric: {metric}. Choose from 'accuracy', 'bacc', 'mf1', or 'auc-roc'.")

        return score
    
    def test(self):
        self.model['default'].eval()

        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                batch = batch_to_gpu(batch, self.device)
                data = batch['data']
                encoder_out = self.model['default'].encode(data.x, data.adj_t, data.batch)
                logits = self.model['default'].cls(encoder_out)

                preds = logits.argmax(dim=-1)
                all_preds.append(preds.cpu().numpy())

                all_labels.append(data.y.cpu().numpy())

                probs = F.softmax(logits, dim=-1)
                all_probs.append(probs.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        all_probs = np.concatenate(all_probs)

        acc = accuracy_score(all_labels, all_preds)
        bacc = balanced_accuracy_score(all_labels, all_preds)
        mf1 = f1_score(all_labels, all_preds, average="macro")

        if self.dataset.num_classes == 2:
            roc = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            roc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')

        return acc, bacc, mf1, roc

def consis_loss(logps, temp=0.5):
    ps = [torch.exp(p) for p in logps]
    sum_p = 0.
    for p in ps:
        sum_p = sum_p + p
    avg_p = sum_p / len(ps)

    sharp_p = (torch.pow(avg_p, 1. / temp) /
               torch.sum(torch.pow(avg_p, 1. / temp), dim=1, keepdim=True)).detach()
    loss = 0.
    for p in ps:
        loss += torch.mean((p - sharp_p).pow(2).sum(1))
    loss = loss / len(ps)
    return 1 * loss


def batch_to_gpu(batch, device):
    for key in batch:
        if isinstance(batch[key], list):
            for i in range(len(batch[key])):
                batch[key][i] = batch[key][i].to(device)
        else:
            batch[key] = batch[key].to(device)

    return batch