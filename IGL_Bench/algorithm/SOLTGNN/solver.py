from torch import optim
from IGL_Bench.algorithm.SOLTGNN.backbone import GIN, GCN
from IGL_Bench.algorithm.SOLTGNN.PatternMemory import PatternMemory
from IGL_Bench.algorithm.SOLTGNN.utils import load_data, load_sample
import torch
import os
import math
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from torch_geometric.loader import DataLoader, RandomNodeSampler


class SOLTGNN_graph_solver:
    def __init__(self, config, dataset, device='cuda'):
        print("------------------------notice------------------------")
        print("To run SOLT-GNN algorithm, you need to unzip the sampling folder in the IGL_Bench/algorithm/SOLTGNN folder.")
        self.config = config
        self.dataset = dataset
        self.device = device

        self.model = {}
        self.optimizer = {}
        self.initialization()

        self.model['default'] = self.model['default'].to(device)

    def initialization(self):
        degree_as_tag = True
        if self.dataset.name == "PROTEINS":
            hidden_dim = 32
            batch_size = 32
            learn_eps = False
            degree_as_tag = False
            weight_decay = 0
        elif self.dataset.name == "PTC_MR":
            hidden_dim = 32
            batch_size = 32
            learn_eps = True
            weight_decay = 5e-4
        elif self.dataset.name == "PTC_MR":
            hidden_dim = 32
            batch_size = 32
            learn_eps = True
            weight_decay = 5e-4
        elif self.dataset.name == "IMDB-BINARY":
            hidden_dim = 32
            batch_size = 32
            learn_eps = True
            weight_decay = 5e-4
        elif self.dataset.name == "DD":
            hidden_dim = 32
            batch_size = 32
            learn_eps = False
            degree_as_tag = False
            weight_decay = 0
        elif self.dataset.name == "FRANKENSTEIN":
            hidden_dim = 32
            batch_size = 32
            learn_eps = True
            weight_decay = 5e-4
        elif self.dataset.name == "REDDIT-BINARY":
            hidden_dim = 32
            batch_size = 32
            learn_eps = False
            weight_decay = 5e-4
        elif self.dataset.name == "COLLAB":
            hidden_dim = 32
            batch_size = 32
            learn_eps = False
            weight_decay = 5e-4
        else:
            raise ValueError("dataset does not exists")

        self.config.hidden_dim = hidden_dim
        self.config.batch_size = batch_size
        self.config.weight_decay = weight_decay
        self.config.learn_eps = learn_eps

        graphs, num_classes = load_data(self.dataset.name, degree_as_tag)

        gsamples = load_sample(self.dataset.name, graphs)

        nodes = torch.zeros(len(graphs))

        for i in range(len(graphs)):
            nodes[i] = graphs[i].g.number_of_nodes()
        _, ind = torch.sort(nodes, descending=True)
        for i in ind[:self.config.K]:
            graphs[i].nodegroup += 1

        train_mask = self.dataset.train_mask
        val_mask = self.dataset.val_mask
        test_mask = self.dataset.test_mask
        self.train_graphs = [g for g, m in zip(graphs, train_mask) if m]
        self.valid_graphs = [g for g, m in zip(graphs, val_mask) if m]
        self.test_graphs = [g for g, m in zip(graphs, test_mask) if m]
        self.train_samples = [g for g, m in zip(gsamples, train_mask) if m]
        in_channels = self.train_graphs[0].node_features.shape[1]
        if self.config.backbone == 'GIN':
            model = GIN(self.config.n_layer, in_channels, self.config.hidden_dim, num_classes, self.config.dropout,
                        self.device, self.config.graph_pooling_type).to(self.device)
        elif self.config.backbone == 'GCN':
            model = GCN(self.config.n_layer, in_channels, self.config.hidden_dim, num_classes, self.config.dropout,
                        self.device, self.config.graph_pooling_type).to(self.device)
        else:
            raise ValueError("Backbone Error")
        self.patmem = PatternMemory(self.config.hidden_dim, self.config.dm).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        self.criterion_ce = torch.nn.CrossEntropyLoss()
        self.criterion_cs = torch.nn.CosineSimilarity(dim=1, eps=1e-7)
        self.model['default'] = model
        self.optimizer['default'] = optimizer
        
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
        self.patmem.reset_parameters()

    def train_one_step(self, config, model, patmem, graphs, samples, optimizer, epoch):
        model.train()
        patmem.train()
        batch_size = config.batch_size
        loss_accum = 0
        shuffle = np.random.permutation(len(graphs))
        train_graphs = [graphs[ind] for ind in shuffle]
        train_samples = [samples[ind] for ind in shuffle]

        idx = np.arange(len(train_graphs))

        l_h = l_t = l_n = l_g = l_d = 0

        for i in range(0, len(train_graphs), batch_size):
            selected_idx = idx[i:i + batch_size]
            if len(selected_idx) == 0:
                continue

            batch_graph_h = [train_graphs[idx] for idx in selected_idx if train_graphs[idx].nodegroup == 1]
            batch_samples_h = [train_samples[idx] for idx in selected_idx if train_graphs[idx].nodegroup == 1]
            batch_graph_t = [train_graphs[idx] for idx in selected_idx if train_graphs[idx].nodegroup == 0]

            n_h = len(batch_graph_h)
            n_t = len(batch_graph_t)
            n = n_h + n_t

            if n_h <= 1 or n_t == 0:
                continue

            embeddings_head = model.get_patterns(batch_graph_h)

            gsize = np.zeros(n_h + 1, dtype=int)

            for i, graph in enumerate(batch_graph_h):
                gsize[i + 1] = gsize[i] + graph.g.number_of_nodes()

            q_idx = []
            pos_idx = []
            neg_idx = []
            pos_rep = []
            neg_rep = []

            for i, graph in enumerate(batch_graph_h):
                graph.sample_list = batch_samples_h[i].sample_list[epoch]
                gsize[i + 1] = gsize[i] + graph.g.number_of_nodes()
                uidx = batch_samples_h[i].unsample_list[epoch] + gsize[i]
                pos_rep.append(embeddings_head[uidx].sum(dim=0, keepdim=True))
                for _ in range(config.n_g):
                    neg = np.random.randint(n_h)
                    while neg == i:
                        neg = np.random.randint(n_h)
                    m = min(len(uidx), batch_graph_h[neg].g.number_of_nodes())
                    sample_idx = torch.tensor(np.random.permutation(
                        batch_graph_h[neg].g.number_of_nodes())).long()
                    sample_idx += gsize[neg]
                    neg_rep.append(embeddings_head[sample_idx[:m]].sum(dim=0, keepdim=True))
                for _ in range(config.n_n):
                    neg = np.random.randint(n_h)
                    while neg == i:
                        neg = np.random.randint(n_h)
                    size = min(batch_graph_h[neg].g.number_of_nodes(), batch_graph_h[i].g.number_of_nodes())
                    q_idx.append(torch.arange(gsize[i], gsize[i] + size).long())
                    sample_idx = torch.tensor(np.random.permutation(graph.g.number_of_nodes())).long()
                    sample_idx += gsize[i]
                    pos_idx.append(sample_idx[:size])
                    sample_idx = torch.tensor(np.random.permutation(batch_graph_h[neg].g.number_of_nodes())).long()
                    sample_idx += gsize[neg]
                    neg_idx.append(sample_idx[:size])

            q_idx = torch.cat(q_idx).long()
            pos_idx = torch.cat(pos_idx).long()
            neg_idx = torch.cat(neg_idx).long()

            query = patmem(embeddings_head[q_idx])
            pos = embeddings_head[pos_idx]
            neg = embeddings_head[neg_idx]

            loss_n = - (torch.mul(query.div(torch.norm(query, dim=1).reshape(-1, 1) + 1e-7),
                                  pos.div(torch.norm(pos, dim=1).reshape(-1, 1) + 1e-7)).sum(dim=1) -
                        torch.mul(query.div(torch.norm(query, dim=1).reshape(-1, 1) + 1e-7),
                                  neg.div(torch.norm(neg, dim=1).reshape(-1, 1) + 1e-7)).sum(
                            dim=1)).sigmoid().log().mean()

            subgraph_rep = model.subgraph_rep(batch_graph_h)
            pos_rep = torch.cat(pos_rep)
            neg_rep = torch.cat(neg_rep)
            query_g = patmem(subgraph_rep).repeat(config.n_g, 1)
            pos_g = pos_rep.repeat(config.n_g, 1)
            neg_g = neg_rep

            loss_g = - (torch.mul(query_g.div(torch.norm(query_g, dim=1).reshape(-1, 1) + 1e-7),
                                  pos_g.div(torch.norm(pos_g, dim=1).reshape(-1, 1) + 1e-7)).sum(dim=1) -
                        torch.mul(query_g.div(torch.norm(query_g, dim=1).reshape(-1, 1) + 1e-7),
                                  neg_g.div(torch.norm(neg_g, dim=1).reshape(-1, 1) + 1e-7)).sum(
                            dim=1)).sigmoid().log().mean()

            graph_repre_head = model.get_graph_repre(batch_graph_h)
            patterns_head = patmem(graph_repre_head)
            output_h = model.predict(graph_repre_head + patterns_head)
            labels_h = torch.LongTensor([graph.label for graph in batch_graph_h]).to(self.device)
            loss_h = self.criterion_ce(output_h, labels_h)

            graph_repre_tail = model.get_graph_repre(batch_graph_t)
            patterns_tail = patmem(graph_repre_tail)
            output_t = model.predict(graph_repre_tail + patterns_tail)
            labels_t = torch.LongTensor([graph.label for graph in batch_graph_t]).to(self.device)
            loss_t = self.criterion_ce(output_t, labels_t)

            loss_d = (self.criterion_cs(graph_repre_tail, patterns_tail).sum() + self.criterion_cs(graph_repre_head,
                                                                                         patterns_head).sum()) / n

            l_t += loss_t.detach().cpu().numpy()
            l_h += loss_h.detach().cpu().numpy()
            l_n += loss_n.detach().cpu().numpy()
            l_g += loss_g.detach().cpu().numpy()
            l_d += loss_d.detach().cpu().numpy()

            loss = 2 * (config.alpha * loss_h + (
                    1 - config.alpha) * loss_t) + config.mu1 * loss_n + config.mu2 * loss_g + config.lbd * loss_d

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = loss.detach().cpu().numpy()
            loss_accum += loss

        return loss_accum

    def train(self):
        self.reset_parameters()
        num_epochs = getattr(self.config, 'epochs', 500)
        patience = getattr(self.config, 'patience', 100)
        model = self.model['default']
        patmem = self.patmem
        optimizer = self.optimizer['default']
        self.patmem.train()
        best_val_accuracy = 0
        epochs_without_improvement = 0

        for epoch in range(num_epochs):
            model.train()
            loss_accum = self.train_one_step(self.config, model, patmem, self.train_graphs, self.train_samples,
                                             optimizer, epoch)
            avg_loss = loss_accum / len(self.train_graphs)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

            val_accuracy, _, _, _ = self.eval(model, patmem, self.valid_graphs)

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch+2}.")
                break

        print("Training Finished!")


    @torch.no_grad()
    def pass_data_iteratively(self, model, patmem, graphs, batch_size=128):
        model.eval()
        patmem.eval()
        output = []
        labels = []
        readout = []
        idx = np.arange(len(graphs))
        for i in range(0, len(graphs), batch_size):
            selected_idx = idx[i:i + batch_size]
            if len(selected_idx) == 0:
                continue
            batch_graph = [graphs[i] for i in selected_idx]

            embeddings_graph = model.get_graph_repre(batch_graph)
            patterns = patmem(embeddings_graph)
            readout.append(embeddings_graph)
            output.append(model.predict(embeddings_graph + patterns))
            labels.append(torch.LongTensor([graph.label for graph in batch_graph]))

        return torch.cat(output, 0), torch.cat(labels, 0).to(self.device), torch.cat(readout, dim=0)

    @torch.no_grad()
    def eval(self, model, patmem, graphs):
        batch_size = self.config.batch_size
        output, labels, readout = self.pass_data_iteratively(model, patmem, graphs, batch_size)
        pred = output.max(1, keepdim=True)[1]
        pred_np = pred.cpu().numpy().flatten()
        labels_np = labels.cpu().numpy().flatten()

        acc = accuracy_score(labels_np, pred_np)
        macro_f1 = f1_score(labels_np, pred_np, average='macro')
        b_acc = balanced_accuracy_score(labels_np, pred_np)
        output_prob = F.softmax(output, dim=1).cpu().numpy()
        num_classes = output_prob.shape[1]
        if num_classes > 2:
            auc_roc = roc_auc_score(labels_np, output_prob, multi_class='ovr')
        else:
            auc_roc = roc_auc_score(labels_np, output_prob[:, 1])
        return acc, b_acc, macro_f1, auc_roc

    @torch.no_grad()
    def test(self):
        model = self.model['default']
        patmem = self.patmem
        graphs = self.test_graphs
        return self.eval(model, patmem, graphs)