from IGL_Bench.algorithm.DPGNN.utils import *
from IGL_Bench.algorithm.DPGNN.model import *
from IGL_Bench.algorithm.DPGNN.learn import *

import torch
import numpy as np
import copy
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

class DPGNN_node_solver:
    def __init__(self, config, dataset, device: str = 'cuda'):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.data = dataset.to(self.device)
        self.num_classes: int = int(self.data.y.max().item() + 1)
        self.num_features: int = self.data.x.size(1)
        self.config.num_classes = self.num_classes
        self.config.num_features = self.num_features
        self.classes = torch.arange(self.num_classes, device=self.device)
        self.config.classes = self.classes
        train_counts = torch.bincount(self.data.y[self.data.train_mask].cpu(),
                                      minlength=self.num_classes)
        self.config.c_train_num = train_counts
        if getattr(self.config, 'ssl', 'no') == 'yes':
            self.config.deg_inv_sqrt = deg(self.data.edge_index, self.data.x).to(self.device)
        if getattr(self.config, 'backbone', 'GCN') == 'GCN':
            self.encoder = GCN(self.config).to(self.device)
        else:
            raise ValueError(f"Unsupported encoder: {self.config.encoder}")
        self.prototype_net = prototype().to(self.device)
        self.dist_encoder = dist_embed(self.config).to(self.device)
        self._build_optimizer()
        self.criterion = torch.nn.NLLLoss()
        self.data.y_aug = self.data.y.clone()

    def _build_optimizer(self):
        param_groups = [
            {'params': self.encoder.conv1.parameters(), 'lr': 1e-2, 'weight_decay': 5e-4},
            {'params': self.encoder.conv2.parameters(), 'lr': 1e-2, 'weight_decay': 0.0},
            {'params': self.dist_encoder.lin.parameters(), 'lr': 1e-2, 'weight_decay': 0.0},
        ]
        self.optimizer = torch.optim.Adam(param_groups)

    def reset_parameters(self):
        self.encoder.conv1.reset_parameters()
        self.encoder.conv2.reset_parameters()
        self.dist_encoder.lin.reset_parameters()

    def _label_prop_augment(self):
        if getattr(self.config, 'label_prop', 'no') != 'yes':
            return
        y_prop = label_prop(self.data.edge_index,
                            self.data.train_mask,
                            self.config.c_train_num,
                            self.data.y,
                            epochs=20)
        y_aug, new_train_mask = sample(self.data.train_mask,
                                       self.config.c_train_num,
                                       y_prop,
                                       self.data.y,
                                       eta=self.config.eta)
        self.data.y_aug = y_aug.to(self.device)
        self.data.train_mask = new_train_mask.to(self.device)

    def train(self):
        self.reset_parameters()
        self._label_prop_augment()
        best_val_f1 = -1.0
        early_stopping = getattr(self.config, 'early_stopping', 10)
        history = []
        for epoch in range(getattr(self.config, 'epochs', 500)):
            train(self.encoder, self.dist_encoder, self.prototype_net,
                  self.data, self.optimizer, self.criterion, self.config)
            f1_all, _, _ = test(self.encoder, self.dist_encoder, self.prototype_net,
                                 self.data, self.config)
            val_f1_mean = np.mean(f1_all[1])
            print('Epoch: {:03d}, val_f1_mean: {:.4f}'.format(epoch, val_f1_mean))
            history.append(val_f1_mean)
            if val_f1_mean > best_val_f1:
                best_val_f1 = val_f1_mean
                self._best_state = {
                    'encoder': copy.deepcopy(self.encoder.state_dict()),
                    'dist': copy.deepcopy(self.dist_encoder.state_dict())
                }
            if early_stopping > 0 and epoch > self.config.epochs // 10:
                if len(history) > early_stopping:
                    recent = np.array(history[-early_stopping:])
                    if val_f1_mean < recent.mean():
                        break
        if hasattr(self, '_best_state'):
            self.encoder.load_state_dict(self._best_state['encoder'])
            self.dist_encoder.load_state_dict(self._best_state['dist'])

    def test(self):
        self.encoder.eval()
        with torch.no_grad():
            embedding = self.encoder(self.data)
            proto_list = []
            for c in self.classes:
                idx = (self.data.y_aug == c) & self.data.train_mask
                proto_list.append(self.prototype_net(embedding[idx]))
            proto = torch.stack(proto_list, dim=0)
            query_emb = embedding[self.data.test_mask]
            query_dist = self.dist_encoder(query_emb, proto, self.classes)
            proto_dist = self.dist_encoder(proto, proto, self.classes)
            logits = torch.log_softmax(torch.mm(query_dist, proto_dist), dim=1)
            probs = torch.exp(logits).cpu().numpy()
            preds = logits.max(dim=1)[1].cpu()
            labels = self.data.y[self.data.test_mask].cpu()
            acc = (preds == labels).sum().item() / labels.size(0)
            bacc = balanced_accuracy_score(labels, preds)
            mf1 = f1_score(labels, preds, average='macro', zero_division=0)
            try:
                roc = roc_auc_score(labels, probs, multi_class='ovr')
            except Exception:
                roc = float('nan')
        return acc, bacc, mf1, roc
