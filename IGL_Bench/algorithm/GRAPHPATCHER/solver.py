from copy import deepcopy

from torch.utils.data import DataLoader
from torch import nn

from IGL_Bench.algorithm.GRAPHPATCHER.GCN import GCN, Generator
from IGL_Bench.algorithm.GRAPHPATCHER.data_load import load_data, preprocess, Graph_Dataset, Graph_Dataset_splits, \
    Graph_Collator_train, Graph_Collator_infer
from IGL_Bench.algorithm.GRAPHPATCHER.utils import inject_nodes, kl_div
from IGL_Bench.backbone.gcn import GCN_node_sparse
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score


class GRAPHPATCHER_node_solver:
    def __init__(self, config, dataset, device='cuda'):
        self.config = config
        self.dataset = dataset
        self.device = device

        self.model = {}
        self.optimizer = {}
        self.initializtion()

        self.model['default'] = self.model['default'].to(device)
        self.dataset = self.dataset.to(self.device)

        self.accuracy = 0.0
        self.macro_f1 = 0.0
        self.bacc = 0.0
        self.auc_roc = 0.0

    def initializtion(self):
        in_size = self.dataset.num_features
        num_classes = self.dataset.y.numpy().max().item() + 1
        out_size = num_classes

        pretrain_args = self.config.pretrain
        generator_args = self.config.generator

        self.model['gcn'] = GCN(in_size, pretrain_args.hid_dim + [out_size], norm=pretrain_args.norm, mp_norm=pretrain_args.mp_norm).to(self.device)

        self.model['generator'] = Generator(generator_args.dropout, self.model['gcn'].hidden_lst[0], generator_args.hid_dim, self.model['gcn'].hidden_lst[0], generator_args,
                          three_layer=generator_args.three_layer, norm=generator_args.norm, mp_norm=generator_args.mp_norm).to(self.device)

        self.optimizer['gcn'] = torch.optim.Adam(self.model['gcn'].parameters(), lr=pretrain_args.lr,
                                                     weight_decay=pretrain_args.weight_decay)

        self.optimizer['generator'] = torch.optim.AdamW(self.model['generator'].parameters(), lr=generator_args.lr, weight_decay=generator_args.weight_decay)

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
        self.pretrain_gnn()

        self.optimizer['gcn'].eval()
        for param in self.optimizer['gcn'].parameters():
            param.requires_grad = False

        generator_args = self.config.generator

        graph = load_data(generator_args.dataset,
                          preprocess_=False if self.dataset.data_name == 'arxiv' else True)
        graph.ndata['train_mask'], graph.ndata['val_mask'], graph.ndata['test_mask'] = (
            self.dataset.train_mask, self.dataset.val_mask, self.dataset.test_mask)
        preprocessed_graph = preprocess(graph)

        dataset = Graph_Dataset(graph, generator_args.degree_train, generator_args.drop_ratio, 10, generator_args.k)
        dataset_val = Graph_Dataset_splits(preprocessed_graph, graph.ndata['val_mask'].nonzero().squeeze(), generator_args.k)
        dataset_test = Graph_Dataset_splits(preprocessed_graph, graph.ndata['test_mask'].nonzero().squeeze(), generator_args.k)
        collator_train = Graph_Collator_train()
        collator_inference = Graph_Collator_infer()

        dataloader = DataLoader(dataset=dataset, drop_last=True, \
                                batch_size=generator_args.batch_size, collate_fn=collator_train, num_workers=generator_args.workers)
        dataloader_val = DataLoader(dataset=dataset_val, shuffle=False, drop_last=False, \
                                    batch_size=generator_args.batch_size * 4, collate_fn=collator_inference,
                                    num_workers=int(generator_args.workers))
        dataloader_test = DataLoader(dataset=dataset_test, shuffle=False, drop_last=False, \
                                     batch_size=generator_args.batch_size * 4, collate_fn=collator_inference,
                                     num_workers=int(generator_args.workers))

        generator = self.model['generator']
        GNN = self.model['gnn']

        accumulate_counter = 0
        accumulated_loss = 0
        accumulated_loss_ = 0
        self.optimizer['generator'].zero_grad()
        best_val_acc = 0
        test_acc = 0
        best_generation_step = 0
        patience = 0
        generator.train()
        current_iteration = 0

        while current_iteration < generator_args.training_iteration:
            # Training Loops
            for data in dataloader if generator_args.bar else dataloader:
                # Starting GraphPatcher Training
                batched_graphs, inverse_indices = data[0], data[1]
                batched_graphs = [bg.to(self.device) for bg in batched_graphs]
                starting_graphs = batched_graphs[0]
                generation_targets = batched_graphs[1:]
                loss = 0
                # Iterative Patching
                for generation_graph, this_inverse_indces in zip(generation_targets, inverse_indices[1:-1]):
                    with torch.no_grad():
                        target_distribution = GNN(generation_graph, generation_graph.ndata['feat'])[this_inverse_indces]
                        target_distribution = target_distribution.reshape(generator_args.batch_size, 10, -1)
                    generated_neighbors = generator(starting_graphs, inverse_indices[0])
                    starting_graphs = inject_nodes(starting_graphs, generated_neighbors, inverse_indices[0], self.device)
                    reconstructed_distribution = GNN(starting_graphs, starting_graphs.ndata['feat'])[inverse_indices[0]]
                    reconstructed_distribution = reconstructed_distribution.unsqueeze(1).expand_as(target_distribution)
                    loss += kl_div(reconstructed_distribution, target_distribution)
                    starting_graphs.ndata['feat'] = starting_graphs.ndata['feat'].detach()

                # Final Patching
                with torch.no_grad():
                    target_distribution = GNN(generation_targets[-1], generation_targets[-1].ndata['feat'])[
                        inverse_indices[-1]]
                generated_neighbors = generator(starting_graphs, inverse_indices[0])
                starting_graphs = inject_nodes(starting_graphs, generated_neighbors, inverse_indices[0], self.device)
                reconstructed_distribution = GNN(starting_graphs, starting_graphs.ndata['feat'])[inverse_indices[0]]
                loss += kl_div(reconstructed_distribution, target_distribution)
                loss.backward()

                accumulate_counter += 1
                accumulated_loss += loss.item()
                accumulated_loss_ += loss.item()

                print(f"Epoch [{current_iteration}/{generator_args.training_iteration}], Loss: {loss.item():.4f}")

                if accumulate_counter % generator_args.accumulate_step == 0:
                    self.optimizer['generator'].step()
                    self.optimizer['generator'].zero_grad()
                    current_iteration += 1
                    if current_iteration % generator_args.eval_iteration == 0:
                        if current_iteration < generator_args.warmup_steps:
                            print(
                                f"[Warming up {current_iteration}/{generator_args.training_iteration}]: accumulated loss {round(accumulated_loss_, 4)}")
                            accumulated_loss_ = 0
                            continue

                        accumulated_loss_ = 0
                        # Validation Loop
                        generator.eval()
                        vals = self.eval(dataloader_val, iteration=generator_args.total_generation_iteration)
                        val_acc, val_entropy, val_f1, val_bacc, val_roc_auc = \
                            ([val[0] for val in vals], [val[1] for val in vals], [val[2] for val in vals],
                             [val[3] for val in vals], [val[4] for val in vals])

                        current_best_generation_step = val_acc.index(
                            max(val_acc)) if generator_args.generation_iteration == -1 else generator_args.generation_iteration - 1

                        current_val_acc, current_val_entropy, current_val_f1, current_val_bacc, current_val_roc_auc = vals[current_best_generation_step]
                        statement = current_val_acc >= best_val_acc

                        if statement:
                            best_val_acc = current_val_acc
                            best_generation_step = current_best_generation_step
                            patience = 0
                            tests = self.eval(dataloader_test, iteration=generator_args.total_generation_iteration)
                            test_acc, _, test_f1, test_bacc, test_roc_auc = tests[best_generation_step]

                            self.accuracy = test_acc
                            self.macro_f1 = test_f1
                            self.bacc = test_bacc
                            self.auc_roc = test_roc_auc

                        else:
                            patience += 1

                        if patience == generator_args.patience:
                            print(f"Early stopping at epoch {current_iteration + 1}.")
                            current_iteration = generator_args.training_iteration
                            break
                        generator.train()

        print("Training Finished!")

    def pretrain_gnn(self):
        loss_fcn = nn.CrossEntropyLoss()
        best_eval, counter, patience, saved_model = 0, 0, 500, None
        best_model_state_dict = None
        labels = self.dataset.y
        print("Pretrain GCN")
        for epoch in range(10000):
            if counter == patience:
                break
            else:
                counter += 1
            self.model['gcn'].train()
            logits = self.model['gcn'](self.dataset.adj, self.dataset.x)
            loss = loss_fcn(logits[self.dataset.train_mask], labels[self.dataset.train_mask].long())
            self.optimizer['gcn'].zero_grad()
            loss.backward()
            self.optimizer['gcn'].step()

            acc = self.eval_pretrain()
            if acc > best_eval:
                best_eval = acc
                counter = 0
                best_model_state_dict = deepcopy(self.model['gcn'])
            print(f"Pretrain Epoch [{epoch}/{10000}], Loss: {loss.item():.4f}")

        self.model['gcn'].load_state_dict(best_model_state_dict)


    def eval_pretrain(self):
        self.model['gcn'].eval()
        all_labels = self.dataset.y[self.dataset.val_mask].cpu().numpy()

        with torch.no_grad():
            out = self.model['gcn'](self.dataset.adj, self.dataset.x)
            predictions = out[self.dataset.val_mask].argmax(dim=1).cpu().numpy()

        return accuracy_score(all_labels, predictions)

    def eval(self, dataloader, metric="accuracy", iteration=1, return_dist=False):
        """ Evaluate the model on the validation or test set using the selected metric. """
        recon_dist_list = {}
        generator = self.model['generator']
        for i in range(iteration): recon_dist_list[i] = []
        generator.eval()
        for _, data in enumerate(dataloader if self.config.generator.bar else dataloader):
            batched_graphs, offset = data
            batched_graphs = batched_graphs.to(self.device)
            for itera in range(iteration):
                with torch.no_grad():
                    generated_neighbors = generator(batched_graphs, offset)
                batched_graphs = inject_nodes(batched_graphs, generated_neighbors, offset, self.device)
                with torch.no_grad():
                    recon_dist_list[itera].append(self.model['gnn'](batched_graphs, batched_graphs.ndata['feat'])[offset].cpu())

        all_labels = self.dataset.y[self.dataset.val_mask].cpu().numpy()

        returns = []
        for i in range(iteration):
            predictions = torch.cat(recon_dist_list[i])
            current_acc = accuracy_score(all_labels, predictions)
            current_f1 = f1_score(all_labels, predictions, average='macro')
            entropy = F.cross_entropy(predictions, all_labels.long())
            current_bacc = balanced_accuracy_score(all_labels, predictions)
            current_roc_auc = roc_auc_score(all_labels, predictions, multi_class='ovr', average='macro')

            if return_dist:
                returns.append([current_acc, entropy.item(), predictions, current_f1, current_bacc, current_roc_auc])
            else:
                returns.append([current_acc, entropy.item(), current_f1, current_bacc, current_roc_auc])

        return returns

    def test(self):
        return self.accuracy, self.bacc, self.macro_f1, self.auc_roc
