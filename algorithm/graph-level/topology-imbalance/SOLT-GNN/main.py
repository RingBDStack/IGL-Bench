import argparse
import csv
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from gcn import GCN
from gin import GIN
from graphsage import GraphSAGE
from PatternMemory import PatternMemory
from util import data_split, load_data, load_sample, tsne_test, load_split
import sys
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score, silhouette_score

criterion_ce = nn.CrossEntropyLoss()
criterion_cs = nn.CosineSimilarity(dim=1, eps=1e-7)


def train_gin(args, model, device, graphs, optimizer, epoch):
    model.train()
    loss_accum = 0
    minibatch_size = args.batch_size
    idx = np.arange(len(graphs))
    shuffle_idx = np.random.permutation(len(graphs))
    train_graphs = [graphs[id] for id in shuffle_idx]
    for i in range(0, len(train_graphs), minibatch_size):
        selected_idx = idx[i:i + minibatch_size]
        if len(selected_idx) == 0:
            continue
        batch_graph_h = [train_graphs[idx] for idx in selected_idx]
        output_h = model(batch_graph_h)
        labels_h = torch.LongTensor([graph.label for graph in batch_graph_h]).to(device)
        loss = criterion_ce(output_h, labels_h)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss = loss.detach().cpu().numpy()
        loss_accum += loss
    average_loss = loss_accum / float(len(train_graphs))
    # print("loss training: %f" % (average_loss), end = ' ')
    return average_loss


@torch.no_grad()
def pass_data_iteratively_gin(model, graphs, minibatch_size=128):
    model.eval()
    output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i + minibatch_size]
        if len(sampled_idx) == 0:
            continue
        output.append(model([graphs[j] for j in sampled_idx]).detach())
    return torch.cat(output, 0)


@torch.no_grad()
def test_gin(args, model, device, graphs, epoch):
    model.eval()
    minibatch_size = args.batch_size
    output = pass_data_iteratively_gin(model, graphs, minibatch_size)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor(
        [graph.label for graph in graphs]).to(device)
    loss = criterion_ce(output, labels)
    correct = pred.eq(labels.view_as(
        pred)).sum().cpu().item()
    acc = correct / len(graphs)
    mask_h = torch.zeros(len(graphs))
    for j in range(len(graphs)):
        mask_h[j] = graphs[j].nodegroup
    mask_h = mask_h.bool()
    mask_t = ~ mask_h
    large_correct = pred[mask_h].eq(labels[mask_h].view_as(
        pred[mask_h])).sum().cpu().item()
    acc_head = large_correct / float(mask_h.sum())
    small_correct = pred[mask_t].eq(labels[mask_t].view_as(
        pred[mask_t])).sum().cpu().item()
    acc_tail = small_correct / float(mask_t.sum())

    return loss, acc, acc_head, acc_tail


def train(args, model, patmem, device, graphs, samples, optimizer, optimizer_p, epoch):
    model.train()
    patmem.train()
    minibatch_size = args.batch_size
    loss_accum = 0
    # print('epoch: %d' % (epoch+1), end=" ")

    shuffle = np.random.permutation(len(graphs))
    train_graphs = [graphs[ind] for ind in shuffle]
    train_samples = [samples[ind] for ind in shuffle]

    idx = np.arange(len(train_graphs))

    l_h = l_t = l_n = l_g = l_d = 0

    for i in range(0, len(train_graphs), minibatch_size):

        selected_idx = idx[i:i + minibatch_size]

        if len(selected_idx) == 0:
            continue

        batch_graph_h = [train_graphs[idx]
                         for idx in selected_idx if train_graphs[idx].nodegroup == 1]
        batch_samples_h = [train_samples[idx]
                           for idx in selected_idx if train_graphs[idx].nodegroup == 1]
        batch_graph_t = [train_graphs[idx]
                         for idx in selected_idx if train_graphs[idx].nodegroup == 0]

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
            for _ in range(args.n_g):
                neg = np.random.randint(n_h)
                while (neg == i):
                    neg = np.random.randint(n_h)
                m = min(len(uidx), batch_graph_h[neg].g.number_of_nodes())
                sample_idx = torch.tensor(np.random.permutation(
                    batch_graph_h[neg].g.number_of_nodes())).long()
                sample_idx += gsize[neg]
                neg_rep.append(embeddings_head[sample_idx[:m]].sum(dim=0, keepdim=True))
            for _ in range(args.n_n):
                neg = np.random.randint(n_h)
                while (neg == i):
                    neg = np.random.randint(n_h)
                size = min(batch_graph_h[neg].g.number_of_nodes(
                ), batch_graph_h[i].g.number_of_nodes())
                q_idx.append(torch.arange(gsize[i], gsize[i] + size).long())
                sample_idx = torch.tensor(np.random.permutation(
                    graph.g.number_of_nodes())).long()
                sample_idx += gsize[i]
                pos_idx.append(sample_idx[:size])
                sample_idx = torch.tensor(np.random.permutation(
                    batch_graph_h[neg].g.number_of_nodes())).long()
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
                              neg.div(torch.norm(neg, dim=1).reshape(-1, 1) + 1e-7)).sum(dim=1)).sigmoid().log().mean()

        subgraph_rep = model.subgraph_rep(batch_graph_h)
        pos_rep = torch.cat(pos_rep)
        neg_rep = torch.cat(neg_rep)
        query_g = patmem(subgraph_rep).repeat(args.n_g, 1)
        pos_g = pos_rep.repeat(args.n_g, 1)
        neg_g = neg_rep

        loss_g = - (torch.mul(query_g.div(torch.norm(query_g, dim=1).reshape(-1, 1) + 1e-7),
                              pos_g.div(torch.norm(pos_g, dim=1).reshape(-1, 1) + 1e-7)).sum(dim=1) -
                    torch.mul(query_g.div(torch.norm(query_g, dim=1).reshape(-1, 1) + 1e-7),
                              neg_g.div(torch.norm(neg_g, dim=1).reshape(-1, 1) + 1e-7)).sum(
                        dim=1)).sigmoid().log().mean()

        graph_repre_head = model.get_graph_repre(batch_graph_h)
        patterns_head = patmem(graph_repre_head)
        output_h = model.predict(graph_repre_head + patterns_head)
        labels_h = torch.LongTensor([graph.label for graph in batch_graph_h]).to(device)
        loss_h = criterion_ce(output_h, labels_h)

        graph_repre_tail = model.get_graph_repre(batch_graph_t)
        patterns_tail = patmem(graph_repre_tail)
        output_t = model.predict(graph_repre_tail + patterns_tail)
        labels_t = torch.LongTensor([graph.label for graph in batch_graph_t]).to(device)
        loss_t = criterion_ce(output_t, labels_t)

        loss_d = (criterion_cs(graph_repre_tail, patterns_tail).sum() + criterion_cs(graph_repre_head,
                                                                                     patterns_head).sum()) / n

        l_t += loss_t.detach().cpu().numpy()
        l_h += loss_h.detach().cpu().numpy()
        l_n += loss_n.detach().cpu().numpy()
        l_g += loss_g.detach().cpu().numpy()
        l_d += loss_d.detach().cpu().numpy()

        loss = 2 * (args.alpha * loss_h + (
                1 - args.alpha) * loss_t) + args.mu1 * loss_n + args.mu2 * loss_g + args.lbd * loss_d

        optimizer.zero_grad()
        optimizer_p.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer_p.step()

        loss = loss.detach().cpu().numpy()
        loss_accum += loss

    print("Loss Training: %f Head: %f Tail: %f Node: %f Graph: %f Dis: %f" %
          (loss_accum, l_t, l_h, l_n, l_g, l_d))

    return loss_accum


@torch.no_grad()
def pass_data_iteratively(args, model, patmem, graphs, device, minibatch_size=128):
    model.eval()
    patmem.eval()
    output = []
    labels = []
    readout = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        selected_idx = idx[i:i + minibatch_size]
        if len(selected_idx) == 0:
            continue
        batch_graph = [graphs[i] for i in selected_idx]

        embeddings_graph = model.get_graph_repre(batch_graph)
        patterns = patmem(embeddings_graph)
        readout.append(embeddings_graph)
        output.append(model.predict(embeddings_graph + patterns))
        labels.append(torch.LongTensor([graph.label for graph in batch_graph]))

    return torch.cat(output, 0), torch.cat(labels, 0).to(device), torch.cat(readout, dim=0)


@torch.no_grad()
def test(args, model, patmem, device, graphs, epoch):
    model.eval()
    minibatch_size = args.batch_size
    output, labels, readout = pass_data_iteratively(
        args, model, patmem, graphs, device, minibatch_size)
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
    test_res = {'Macro-F1': macro_f1, 'B-Acc': b_acc, 'Acc': acc, 'AUC-ROC': auc_roc}
    mask_t = torch.zeros(len(graphs))
    for j in range(len(graphs)):
        mask_t[j] = 1 - graphs[j].nodegroup
    mask_t = mask_t.bool()
    mask_h = ~ mask_t
    loss = criterion_ce(output[mask_t], labels[mask_t])
    pred_large = pred[mask_h].cpu().numpy().flatten()
    labels_large = labels[mask_h].cpu().numpy().flatten()
    acc_head = accuracy_score(labels_large, pred_large)
    f1_macro_head = f1_score(labels_large, pred_large, labels=np.arange(
        0, 2), average=None, zero_division=0)
    f1_micro_head = f1_score(labels_large, pred_large, labels=np.arange(
        0, 2), average='micro', zero_division=0)
    large_res = {'F1-macro': np.mean(f1_macro_head), 'F1-micro': np.mean(f1_micro_head), 'accuracy': acc_head}

    pred_small = pred[mask_t].cpu().numpy().flatten()
    labels_small = labels[mask_t].cpu().numpy().flatten()
    # print(pred_small)
    # print(labels_small)
    # 计算准确率
    acc_tail = accuracy_score(labels_small, pred_small)
    f1_macro_tail = f1_score(labels_small, pred_small, labels=np.arange(
        0, 2), average=None, zero_division=0)
    f1_micro_tail = f1_score(labels_small, pred_small, labels=np.arange(
        0, 2), average='micro', zero_division=0)
    small_res = {'F1-macro': np.mean(f1_macro_tail), 'F1-micro': np.mean(f1_micro_tail), 'accuracy': acc_tail}
    return loss, test_res, large_res, small_res


@torch.no_grad()
def calculate_silhouette_score(args, model, patmem, graphs, device, seed):
    model.eval()
    minibatch_size = args.batch_size
    output, labels, readout = pass_data_iteratively(
        args, model, patmem, graphs, device, minibatch_size)
    pred = output.max(1, keepdim=True)[1]
    pred_np = pred.cpu().numpy().flatten()
    labels_np = labels.cpu().numpy().flatten()
    readout_np = readout.cpu().numpy()
    # 分类别保存嵌入
    unique_classes = np.unique(labels_np)
    for cls in unique_classes:
        cls_embeddings = readout_np[pred_np == cls]
        os.makedirs(f'embeddings/{args.dataset}/', exist_ok=True)
        np.save(f'embeddings/{args.dataset}/class_{cls}_{seed}.npy', cls_embeddings)

    # 计算轮廓系数
    silhouette_avg = silhouette_score(readout_np, labels_np)
    return silhouette_avg


def main():
    # Note: Hyper-parameters need to be tuned in order to obtain results reported in the paper.
    parser = argparse.ArgumentParser(
        description='PyTorch graph convolutional neural net for whole-graph classification')
    parser.add_argument('--dataset', type=str, default="PTC-MR",
                        help='name of dataset (default: PTC-MR)')
    parser.add_argument('--device', type=int, default=0,
                        help='which g pu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of epochs to train (default: 500)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for splitting the dataset')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='test data ratio')
    parser.add_argument('--valid_ratio', type=float, default=0.1,
                        help='valid data ratio')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='number of layers INCLUDING the input one (default: 5)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=32,
                        help='number of hidden units (default: 32)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--graph_pooling_type', type=str, default="sum", choices=["sum", "average"],
                        help='Pooling for over nodes in a graph: sum or average')
    parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "average", "max"],
                        help='Pooling for over neighboring nodes: sum, average or max')
    parser.add_argument('--learn_eps', action="store_true",
                        help='Whether to learn the epsilon weighting for the center nodes. Does not affect training accuracy though.')
    parser.add_argument('--degree_as_tag', action="store_true",
                        help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')
    parser.add_argument('--l2', type=float, default=5e-4,
                        help='the weight decay of adam optimizer')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help=r'weight of head graph classification loss($\alpha $ in the paper)')
    parser.add_argument('--mu1', type=float, default=1.0,
                        help='weight of node-level co-occurrence loss($\mu_1$ in the paper)')
    parser.add_argument('--mu2', type=float, default=1.0,
                        help='weight of subgraph-level co-occurrence loss($\mu_2$ in the paper)')
    parser.add_argument('--lbd', type=float, default=1e-4,
                        help='weight of dissimilarity regularization loss($\lambda $ in the paper)')
    parser.add_argument('--dm', type=int, default=64,
                        help='dimension of pattern memory($d_m $ in the paper)')
    parser.add_argument('--K', type=int, default=72,
                        help='the number of head graphs($K $ in the paper)')
    parser.add_argument('--n_n', type=int, default=1,
                        help='the number of node-level co-occurrence triplets per node at single epoch')
    parser.add_argument('--n_g', type=int, default=1,
                        help='the number of subgraph-level co-occurrence triplets per graph at single epoch')
    parser.add_argument('--split_mode', type=str, default='-1',
                        help='Setting Options for Split, "low", "medium", "high"')
    parser.add_argument('--backbone', type=str, default="GIN", help='name of backbone (default: GIN)')
    parser.add_argument('--silhouette', dest='silhouette', action='store_true', help='Set flag to True')
    parser.set_defaults(silhouette=False)
    args = parser.parse_args()
    # with open('result/' + args.dataset + '.log', 'w+') as f:
    #     # 保存原始的标准输出
    #     original_stdout = sys.stdout
    #
    #     # 将标准输出重定向到文件
    #     sys.stdout = f

    seed = 0
    degree_state = True
    # Base Hyper-parameter configuration
    if args.dataset == "PROTEINS":
        hidden_dim = 32
        batch_size = 32
        seed = 2022
        learn_eps = False
        degree_state = False
        l2 = 0
    elif args.dataset == "PTC":
        hidden_dim = 32
        batch_size = 32
        seed = 0
        learn_eps = True
        l2 = 5e-4
    elif args.dataset == "PTC_MR":
        hidden_dim = 32
        batch_size = 32
        seed = 0
        learn_eps = True
        l2 = 5e-4
    elif args.dataset == "IMDB-BINARY":
        hidden_dim = 32
        batch_size = 32
        seed = 2020
        learn_eps = True
        l2 = 5e-4
    elif args.dataset == "DD":
        hidden_dim = 32
        batch_size = 32
        seed = 2022
        learn_eps = False
        degree_state = False
        l2 = 0
    elif args.dataset == "FRANK":
        hidden_dim = 32
        batch_size = 32
        learn_eps = True
        l2 = 5e-4
        seed = 0
    elif args.dataset == "REDDIT":
        hidden_dim = 32
        batch_size = 32
        learn_eps = False
        l2 = 5e-4
        seed = 0
    elif args.dataset == "COLLAB":
        hidden_dim = 32
        batch_size = 32
        learn_eps = False
        l2 = 5e-4
        seed = 0
    else:
        raise ValueError("dataset does not exists")

    args.hidden_dim = hidden_dim
    args.batch_size = batch_size
    args.l2 = l2
    args.learn_eps = learn_eps
    args.seed = seed

    graphs, num_classes = load_data(args.dataset, degree_state)

    gsamples = load_sample(args.dataset)

    nodes = torch.zeros(len(graphs))

    for i in range(len(graphs)):
        nodes[i] = graphs[i].g.number_of_nodes()

    _, ind = torch.sort(nodes, descending=True)

    for i in ind[:args.K]:
        graphs[i].nodegroup += 1

    times = 5
    acc = np.zeros(times, dtype=float)
    macro_f1 = np.zeros(times, dtype=float)
    b_acc = np.zeros(times, dtype=float)
    auc_roc = np.zeros(times, dtype=float)
    train_times = np.zeros(times, dtype=float)
    silhouette = np.zeros(times, dtype=float)

    large_acc = np.zeros(times, dtype=float)
    small_acc = np.zeros(times, dtype=float)
    large_macro = np.zeros(times, dtype=float)
    small_macro = np.zeros(times, dtype=float)
    large_micro = np.zeros(times, dtype=float)
    small_micro = np.zeros(times, dtype=float)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    start_time = time.time()
    for seed in range(times):
        print('Seed:', seed)
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
        if args.split_mode == '-1':
            train_graphs, valid_graphs, test_graphs, train_samples = data_split(graphs, gsamples, args.valid_ratio,
                                                                                args.test_ratio, seed)
        else:
            train_mask, val_mask, test_mask, boundary_size = load_split(load_path='dataset/' + args.dataset,
                                                                        split_mode=args.split_mode)
            train_graphs = [g for g, m in zip(graphs, train_mask) if m]
            valid_graphs = [g for g, m in zip(graphs, val_mask) if m]
            test_graphs = [g for g, m in zip(graphs, test_mask) if m]
            train_samples = [g for g, m in zip(gsamples, train_mask) if m]
        cnt_node = torch.zeros(2)

        for i in range(len(test_graphs)):
            cnt_node[test_graphs[i].nodegroup] += 1

        if args.backbone == 'GIN':
            model = GIN(args.num_layers, args.num_mlp_layers, train_graphs[0].node_features.shape[1], args.hidden_dim,
                        num_classes,
                        args.dropout, args.learn_eps, args.graph_pooling_type, args.neighbor_pooling_type, device).to(
                device)
        elif args.backbone == 'GCN':
            model = GCN(args.num_layers, train_graphs[0].node_features.shape[1], args.hidden_dim,
                        num_classes, args.dropout, device).to(device)
        elif args.backbone == 'GraphSAGE':
            model = GraphSAGE(args.num_layers, train_graphs[0].node_features.shape[1], args.hidden_dim,
                              num_classes, args.dropout, device).to(device)
        else:
            raise ValueError("Backbone Error")
        patmem = PatternMemory(args.hidden_dim, args.dm).to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

        opt_p = optim.Adam(patmem.parameters(), lr=args.lr, weight_decay=args.l2)

        test_res = 0
        large_res = 0
        small_res = 0
        silhouette_avg = 0.0
        best_valid_acc = 0
        best_valid_loss = 100000
        patience = 0
        train_start_time = time.time()
        for epoch in range(0, args.epochs):

            _ = train(args, model, patmem, device, train_graphs, train_samples, optimizer, opt_p, epoch)

            loss_valid, acc_valid, _, _ = test(args, model, patmem, device, valid_graphs, epoch)

            # print("valid loss: %.4f acc: %.4f" % (loss_valid, acc_valid))

            if loss_valid < best_valid_loss and acc_valid['Acc'] > best_valid_acc:
                best_valid_acc = acc_valid['Acc']
                best_valid_loss = loss_valid
                patience = 0
                _, test_eval, large_eval, small_eval = test(args, model, patmem, device, test_graphs, epoch)
                if args.silhouette:
                    silhouette_avg = calculate_silhouette_score(args, model, patmem, test_graphs, device, seed)
            else:
                patience += 1
            scheduler.step()

            if patience == 100:
                break
        train_end_time = time.time()
        print("Seed: %.4f valid acc: %.4f test acc: %.4f head acc: %.4f tail acc: %.4f" % (
            seed, best_valid_acc, test_eval['Acc'], large_eval['accuracy'], small_eval['accuracy']))
        acc[seed] = test_eval['Acc']
        macro_f1[seed] = test_eval['Macro-F1']
        b_acc[seed] = test_eval['B-Acc']
        auc_roc[seed] = test_eval['AUC-ROC']
        train_times[seed] = train_end_time - train_start_time
        silhouette[seed] = silhouette_avg
        large_acc[seed] = large_eval['accuracy']
        small_acc[seed] = small_eval['accuracy']
        large_macro[seed] = large_eval['F1-macro']
        small_macro[seed] = small_eval['F1-macro']
        large_micro[seed] = large_eval['F1-micro']
        small_micro[seed] = small_eval['F1-micro']

    print('========== SOLT GIN === ' + args.dataset + ' ==========')
    print('alpha: ', args.alpha, ' mu1: ', args.mu1, ' mu2: ', args.mu2)
    res_acc = '{:.2f}({:.2f})'.format(np.mean(acc) * 100, np.std(acc) * 100)
    res_macro_f1 = '{:.2f}({:.2f})'.format(np.mean(macro_f1) * 100, np.std(macro_f1) * 100)
    res_b_acc = '{:.2f}({:.2f})'.format(np.mean(b_acc) * 100, np.std(b_acc) * 100)
    res_auc_roc = '{:.2f}({:.2f})'.format(np.mean(auc_roc) * 100, np.std(auc_roc) * 100)
    res_train_time = '{:.2f}'.format(np.mean(train_times))
    res_silhouette = '{:.4f}'.format(np.mean(silhouette))
    max_memory_allocated = torch.cuda.max_memory_allocated(args.device)
    res_memory = '{:.2f}'.format(max_memory_allocated / (1024 ** 2))
    print('Split Mode:', args.split_mode)
    print('  Accuracy:', res_acc)
    print('B-Accuracy:', res_b_acc)
    print('  Macro-F1:', res_macro_f1)
    print('   AUC-ROC:', res_auc_roc)
    print('Silhouette:', res_silhouette)
    print('Train Time:', res_train_time, '(s)')
    print('GPU Memory:', res_memory, '(MB)')
    print(acc)
    print(large_acc)
    print(small_acc)
    print(silhouette)
    # print('Large Acc: {:.2f}±{:.2f}'.format(np.mean(large_acc) * 100, np.std(large_acc) * 100))
    # print('Small Acc: {:.2f}±{:.2f}'.format(np.mean(small_acc) * 100, np.std(small_acc) * 100))
    # print('Large F1-micro: {:.2f}±{:.2f}'.format(np.mean(large_micro) * 100, np.std(large_micro) * 100))
    # print('Small F1-micro: {:.2f}±{:.2f}'.format(np.mean(small_micro) * 100, np.std(small_micro) * 100))
    # print('Large F1-macro: {:.2f}±{:.2f}'.format(np.mean(large_macro) * 100, np.std(large_macro) * 100))
    # print('Small F1-macro: {:.2f}±{:.2f}'.format(np.mean(small_macro) * 100, np.std(small_macro) * 100))
    end_time = time.time()
    print('cost time: {:.2f}s'.format(end_time - start_time))
    header = ['split_mode', 'acc', 'b-acc', 'macro-f1', 'auc-roc', 'time', 'memory']
    new_experiment_result = [args.split_mode, res_acc, res_b_acc, res_macro_f1, res_auc_roc, res_train_time, res_memory]
    csv_file_path = "result/" + args.backbone + "/" + args.dataset + ".csv"
    file_exists = os.path.exists(csv_file_path)
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(new_experiment_result)


if __name__ == '__main__':
    main()
