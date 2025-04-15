import pickle

from IGL_Bench.algorithm.DEMONet.models import BaseModel
from IGL_Bench.algorithm.GRAPHPATCHER.utils import construct_placeholder
from IGL_Bench.backbone.gcn import GCN_node_sparse
from IGL_Bench.algorithm.TOPOAUC.myloss import ELossFN
from IGL_Bench.algorithm.TOPOAUC.cal import compute_ppr_and_gpr
from IGL_Bench.algorithm.TOPOAUC.util import *
import torch
import tensorflow as tf
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score


class DEMONet_node_solver:
    def __init__(self, config, dataset, device='cuda'):
        self.config = config
        self.dataset = dataset
        self.device = device

        self.model = {}
        self.optimizer = {}
        self.placeholders = None
        self.ppr, self.gpr = compute_ppr_and_gpr(self.dataset, self.config.pagerank_prob)
        self.initializtion()

        self.model['default'] = self.model['default'].to(device)
        self.dataset = self.dataset.to(device)

        self.accuracy = 0.0
        self.macro_f1 = 0.0
        self.bacc = 0.0
        self.auc_roc = 0.0

    def initializtion(self):
        num_classes = self.dataset.y.numpy().max().item() + 1
        self.placeholders = construct_placeholder(self.dataset.num_nodes, self.dataset.num_features, num_classes)
        args = self.config
        degreeTasks, neighbor_list = self.preprocess_data()

        self.model['default'] = BaseModel(self.placeholders, degreeTasks, neighbor_list, num_classes, self.dataset.num_features, hash_dim=args.hash_dim,
                        hidden_dim=args.hidden_dim, num_hash=args.n_hash_kernel, num_layers=args.n_layers)

    def preprocess_data(self):
        degreeNode = np.sum(self.dataset.adj, axis=1).A1
        degreeNode = degreeNode.astype(np.int32)
        degreeValues = set(degreeNode)

        if self.dataset.data_name == 'arxiv':
            neighbor_list = []
            degreeTasks = []
            adj = self.dataset.adj.todense()
            for value in degreeValues:
                degreePosition = [int(i) for i, v in enumerate(degreeNode) if v == value]
                degreeTasks.append((value, degreePosition))

            file_path = './arxiv.pickle'
            with open(file_path, 'rb') as f:
                neighbor_list = pickle.load(f)

        else:
            neighbor_list = []
            degreeTasks = []
            adj = self.dataset.adj.todense()
            for value in degreeValues:
                degreePosition = [int(i) for i, v in enumerate(degreeNode) if v == value]
                degreeTasks.append((value, degreePosition))

                d_list = []
                for idx in degreePosition:
                    neighs = [int(i) for i in range(adj.shape[0]) if adj[idx, i] > 0]
                    d_list += neighs
                neighbor_list.append(d_list)
                assert len(d_list) == value * len(degreePosition), 'The neighbor lists are wrong!'

        return degreeTasks, neighbor_list


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
        model = self.model['default']
        num_classes = self.dataset.y.numpy().max().item() + 1
        placeholders = self.placeholders

        num_epochs = getattr(self.config, 'epoch', 500)
        patience = getattr(self.config, 'patience', 10)

        logits = model.inference()
        log_resh = tf.reshape(logits, [-1, num_classes])
        lab_resh = tf.reshape(placeholders['labels'], [-1, num_classes])
        msk_resh = tf.reshape(placeholders['masks'], [-1])
        loss = model.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
        accuracy = model.masked_accuracy(log_resh, lab_resh, msk_resh)
        find_logits = model.find_logits(log_resh, lab_resh, msk_resh)
        find_label = model.find_label(log_resh, lab_resh, msk_resh)
        find_mask = model.find_mask(log_resh, lab_resh, msk_resh)

        train_op = model.training(loss, lr=self.config.lr, l2_coef=self.config.weight_decay)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer())

        vloss_min = np.inf
        vacc_max = 0.0
        curr_step = 0

        with tf.Session() as sess:
            sess.run(init_op)
            vacc_early_model = 0.0
            vlss_early_model = 0.0

            for epoch in range(num_epochs):
                train_feed_dict = {}
                train_feed_dict.update({placeholders['labels']: self.dataset.y[self.dataset.train_mask]})
                train_feed_dict.update({placeholders['features']: self.dataset.x})
                train_feed_dict.update({placeholders['dropout']: self.config.dropout})
                train_feed_dict.update({placeholders['masks']: self.dataset.train_mask})
                _, loss_value_tr, acc_tr, temp_logits, temp_labels, temp_mask = sess.run(
                    [train_op, loss, accuracy, find_logits, find_label, find_mask], feed_dict=train_feed_dict)

                val_feed_dict = {}
                val_feed_dict.update({placeholders['labels']: self.dataset.y[self.dataset.val_mask]})
                val_feed_dict.update({placeholders['features']: self.dataset.x})
                val_feed_dict.update({placeholders['dropout']: 0.0})
                val_feed_dict.update({placeholders['masks']: self.dataset.val_mask})
                loss_value_val, acc_val, temp_logits, temp_labels, temp_mask = sess.run(
                    [loss, accuracy, find_logits, find_label, find_mask], feed_dict=val_feed_dict)

                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss_value_tr.item():.4f}")

                if acc_val >= vacc_max or loss_value_val <= vloss_min:
                    if acc_val >= vacc_max and loss_value_val <= vloss_min:
                        vacc_early_model = acc_val
                        vlss_early_model = loss_value_val
                    vacc_max = np.max((acc_val, vacc_max))
                    vloss_min = np.min((loss_value_val, vloss_min))
                    curr_step = 0

                    test_feed_dict = {}
                    test_feed_dict.update({placeholders['labels']: self.dataset.y[self.dataset.test_mask]})
                    test_feed_dict.update({placeholders['features']: self.dataset.x})
                    test_feed_dict.update({placeholders['dropout']: 0.0})
                    test_feed_dict.update({placeholders['masks']: self.dataset.test_mask})
                    best_loss_value_test, best_acc_test, temp_logits, temp_labels, temp_mask = sess.run(
                        [loss, accuracy, find_logits, find_label, find_mask], feed_dict=test_feed_dict)

                    temp_logits = temp_logits[self.dataset.test_mask]
                    temp_labels = temp_labels[self.dataset.test_mask]
                    indices = np.argmax(temp_logits, axis=1)
                    one_label = np.argmax(temp_labels, axis=1)
                    best_f1 = f1_score(one_label, indices, average='macro')
                    best_bacc = balanced_accuracy_score(one_label, indices)
                    best_roc_auc = roc_auc_score(temp_labels, temp_logits, multi_class="ovo")

                else:
                    curr_step += 1
                    if curr_step == patience:
                        print(f"Early stopping at epoch {epoch + 1}.")
                        break

            self.accuracy = best_acc_test
            self.macro_f1 = best_f1
            self.bacc = best_bacc,
            self.auc_roc = best_roc_auc

            sess.close()

        print("Training Finished!")

    def eval(self, metric="accuracy"):
        return

    def test(self):
        return self.accuracy, self.bacc, self.macro_f1, self.auc_roc
