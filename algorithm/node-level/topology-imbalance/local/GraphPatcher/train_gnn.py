import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

from src.data import load_data
from src.model import GCN
from src.utils import evaluate_gnn, init_logger, save_model
from src.option import OptionsGNN
from copy import deepcopy
import timeit

def train(g, features, labels, masks, model, args):
    # define train/val samples, loss function and optimizer
    train_mask = masks[0]
    val_mask = masks[1]
    test_mask = masks[2]
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_eval, counter, patience, saved_model = 0, 0, 500, None
    # training loop
    for epoch in range(10000):
        if counter == patience:
            logger.info('Early Stopping...')
            break
        else:
            counter += 1
        model.train()
        logits = model(g, features) 
        loss = loss_fcn(logits[train_mask], labels[train_mask].long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc, _ = evaluate_gnn(g, features, labels, val_mask, model)
        if acc > best_eval:
            best_eval = acc
            counter = 0
            test_acc, f1 = evaluate_gnn(g, features, labels, test_mask, model, '{}/{}.pkl'.format(args.save_dir, args.dataset))
            saved_model = deepcopy(model)
            logger.info(
                "Epoch {:05d} | Loss {:.4f} | val acc {:.4f} |  test acc {:.4f}, test f1 {:.4f}".format(
                    epoch, loss.item(), acc, test_acc, f1
                )
            )
    return test_acc, saved_model


if __name__ == "__main__":
    option = OptionsGNN()
    args = option.parse()
    print(args)
    if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
    logger = init_logger('{}/{}_run.log'.format(args.save_dir, args.dataset))
    g = load_data(args.dataset, t=args.type, split='public', preprocess_=False if args.dataset=='arxiv' else True)
    device = torch.device("cuda:{}".format(args.device))
    g = g.int().to(device)
    features = g.ndata["feat"]
    labels = g.ndata["label"] if args.dataset != 'yelp' else g.ndata["label"].float()
    masks = g.ndata["train_mask"], g.ndata["val_mask"], g.ndata["test_mask"]
    g_iso = dgl.graph((torch.arange(g.num_nodes()), torch.arange(g.num_nodes())), num_nodes=g.num_nodes()).to(device)

    # create GCN model
    in_size = features.shape[1]
    out_size = int(labels.max() + 1) if args.dataset != 'elliptic' else 2
    if args.hid_dim[0] == -1: args.hid_dim = []
    model = GCN(in_size, args.hid_dim + [out_size], norm=args.norm, mp_norm=args.mp_norm).to(device)
    print(model)
    # model training
    logger.info("Training...")
    time = timeit.default_timer()
    test_acc, saved_model = train(g, features, labels, masks, model, args)
    logger.info("Test accuracy {:.2f}| Time Consumed: {:.2f} s".format(test_acc*100, timeit.default_timer()-time))
    save_model(saved_model.state_dict(), 
               {'in_feats':in_size, 
                'hidden_lst':args.hid_dim + [out_size],
                'norm':args.norm,
                'mp_norm':args.mp_norm},
               '{}/{}_{}'.format(args.save_dir, args.dataset, args.type))