from IGL_Bench.algorithm.ImGKB.dataloader import *
from IGL_Bench.algorithm.ImGKB.util import *
from IGL_Bench.algorithm.ImGKB.model import KGIB
from IGL_Bench.algorithm.ImGKB.loss import *
import torch.nn as nn

class ImGKB_graph_solver:
    def __init__(self, config, dataset, device='cuda'):
        self.config = config
        self.dataset = dataset
        self.device = device
        
        adj, features, labels = my_load_data(dataset)
        self.train_loader = GraphBatchGenerator(config, adj, features, labels, dataset.train_index)
        self.val_loader = GraphBatchGenerator(config, adj, features, labels, dataset.val_index)
        self.test_loader = GraphBatchGenerator(config, adj, features, labels, dataset.test_index)
        
        self.model = {}
        self.optimizer = {}
        self.initializtion()
        self.criterion = Loss()
        
    def initializtion(self):
        self.model['KGIB'] = KGIB(input_dim=self.dataset.num_features, 
                                  hidden_dim=self.config.hidden_dim, 
                                  hidden_graphs=self.config.hidden_graphs, 
                                  size_hidden_graphs=self.config.size_hidden_graphs, 
                                  nclass=self.dataset.num_classes, 
                                  max_step=self.config.max_step, 
                                  num_layers=self.config.n_layer, 
                                  backbone=self.config.backbone)
        self.model['KGIB'].to(self.device)
        
        self.optimizer['KGIB'] = torch.optim.Adam(self.model['KGIB'].parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        
    def reset_parameters(self):
        self.initializtion()
        
    def train(self):
        self.reset_parameters()
        
        num_epochs = getattr(self.config, 'epoch', 500)
        patience = getattr(self.config, 'patience', 10)
        least_epoch = getattr(self.config, 'least_epoch', 200)
        
        best_score = None
        patience_counter = 0
        
        for epoch in range(1, num_epochs + 1):
            self.model['KGIB'].train()
            train_loss = AverageMeter()
            train_auc = AverageMeter()
            
            for i in range(self.train_loader.n_batches):
                output, loss = self.train_epoch(self.train_loader.adj[i], self.train_loader.features[i], 
                                     self.train_loader.graph_indicator[i],self.train_loader.y[i], 
                                     self.model['KGIB'], self.optimizer['KGIB'], 
                                     self.criterion, self.config.beta)
                train_loss.update(loss.item(), output.size(0))
                auc_score, _ = Roc_F(output, self.train_loader.y[i])
                train_auc.update(auc_score,output.size(0))
            
            valid_score = self.eval()  
            print(f"Epoch {epoch}/{num_epochs}, Train Loss: {train_loss.avg:.4f}, Train AUC: {train_auc.avg:.4f}, Validation ACC: {valid_score:.4f}")
            
            if best_score is None or valid_score > best_score:
                best_score = valid_score
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience and epoch >= least_epoch:
                    print(f"Early stopping at Epoch {epoch}.")
                    break   
                
        print('Training finished!')
                         
            
    def train_epoch(self, adj, features, graph_indicator, y, model, optimizer, criterion, beta):
        optimizer.zero_grad()
        output, loss_mi = model(adj, features, graph_indicator)
        loss_train =  criterion(output, y)
        loss_train = loss_train + beta * loss_mi
        loss_train.backward()
        optimizer.step()
        return output, loss_train        
    
    def eval(self, score='acc'):
        loader = self.val_loader
        self.model['KGIB'].eval()
        all_logits = []
        all_labels = []

        for i in range(loader.n_batches):
            logits,_ = self.model['KGIB'](loader.adj[i], loader.features[i], loader.graph_indicator[i])
            all_logits.append(logits)
            all_labels.append(loader.y[i])

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        if score == 'acc':
            score = accuracy_score(all_labels.detach().cpu(), torch.argmax(all_logits, dim=-1).detach().cpu())
        elif score == 'bacc':
            score = balanced_accuracy_score(all_labels.detach().cpu(), torch.argmax(all_logits, dim=-1).detach().cpu())
        elif score == 'mf1':
            score = f1_score(all_labels.detach().cpu(), torch.argmax(all_logits, dim=-1).detach().cpu(), average='macro')
        else:
            raise ValueError(f"Unsupported score type: {score}")

        return score

    def test(self):
        self.model['KGIB'].eval()
        all_logits = []
        all_labels = []
        
        for i in range(self.test_loader.n_batches):
            logits,_ = self.model['KGIB'](self.test_loader.adj[i], self.test_loader.features[i], self.test_loader.graph_indicator[i])
            all_logits.append(logits)
            all_labels.append(self.test_loader.y[i])

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        acc = accuracy_score(all_labels.detach().cpu(), torch.argmax(all_logits, dim=-1).detach().cpu())
        bacc = balanced_accuracy_score(all_labels.detach().cpu(), torch.argmax(all_logits, dim=-1).detach().cpu())
        mf1 = f1_score(all_labels.detach().cpu(), torch.argmax(all_logits, dim=-1).detach().cpu(), average='macro')

        if all_labels.max() > 1:  # Multi-class ROC AUC
            roc = roc_auc_score(all_labels.detach().cpu(), F.softmax(all_logits, dim=-1).detach().cpu(), average='macro', multi_class='ovr')
        else:  # Binary ROC AUC
            roc = roc_auc_score(all_labels.detach().cpu(), F.softmax(all_logits, dim=-1)[:,1].detach().cpu(), average='macro')


        return acc, bacc, mf1, roc