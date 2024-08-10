import torch
import dgl
import os
import pickle


def preprocess(graph):
    graph = dgl.to_bidirected(graph, copy_ndata=True)
    return graph


def cross_validation_gen(y, k_fold=5):
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=k_fold)
    train_splits = []
    val_splits = []
    test_splits = []

    for larger_group, smaller_group in skf.split(y, y):
        train_y = y[smaller_group]
        sub_skf = StratifiedKFold(n_splits=2)
        train_split, val_split = next(iter(sub_skf.split(train_y, train_y)))
        train = torch.zeros_like(y, dtype=torch.bool)
        train[smaller_group[train_split]] = True
        val = torch.zeros_like(y, dtype=torch.bool)
        val[smaller_group[val_split]] = True
        test = torch.zeros_like(y, dtype=torch.bool)
        test[larger_group] = True
        train_splits.append(train.unsqueeze(1))
        val_splits.append(val.unsqueeze(1))
        test_splits.append(test.unsqueeze(1))

    return torch.cat(train_splits, dim=1), torch.cat(val_splits, dim=1), torch.cat(test_splits, dim=1)


def load_data(data_name, t='mid', split='random', hetero_graph_path='new_data', preprocess_=True):
    dataset = data_name

    masks = torch.load(f'../data_mask/{dataset}_{t}_.pth')
    print(dataset)
    if dataset == 'cora':
        dataset = dgl.data.CoraGraphDataset()
        g = dataset[0]
        non_directed = True
        g = process_data(g=g, masks=masks, non_directed=non_directed)
    elif dataset == 'citeseer':
        dataset = dgl.data.CiteseerGraphDataset()
        g = dataset[0]
        non_directed = True
        g = process_data(g=g, masks=masks, non_directed=non_directed)
    elif dataset == 'pubmed':
        dataset = dgl.data.PubmedGraphDataset()
        g = dataset[0]
        non_directed = True
        g = process_data(g=g, masks=masks, non_directed=non_directed)
    elif dataset == 'squirrel':
        dataset = dgl.data.SquirrelDataset()
        g = dataset[0]
        non_directed = False
        g = process_data(g=g, masks=masks, non_directed=non_directed)
    elif dataset == 'chameleon':
        dataset = dgl.data.ChameleonDataset()
        g = dataset[0]
        non_directed = False
        g = process_data(g=g, masks=masks, non_directed=non_directed)
    elif dataset == 'actor':
        dataset = dgl.data.ActorDataset()
        g = dataset[0]
        non_directed = False
        g = process_data(g=g, masks=masks, non_directed=non_directed)
    elif dataset == 'computer':
        dataset = dgl.data.AmazonCoBuyComputerDataset()
        g = dataset[0]
        non_directed = True
        g = process_data(g=g, masks=masks, non_directed=non_directed)
    elif dataset == 'photo':
        dataset = dgl.data.AmazonCoBuyPhotoDataset()
        g = dataset[0]
        non_directed = True
        g = process_data(g=g, masks=masks, non_directed=non_directed)
    elif dataset == 'arxiv':
        from ogb.nodeproppred import DglNodePropPredDataset
        dataset = DglNodePropPredDataset(name='ogbn-arxiv')
        g = dataset[0][0]
        g.ndata['label'] = dataset[0][1].squeeze()
        from sklearn.preprocessing import StandardScaler
        norm = StandardScaler()
        norm.fit(g.ndata['feat'])
        g.ndata['feat'] = torch.tensor(norm.transform(g.ndata['feat'])).float()
        non_directed = True
        g = process_data(g=g, masks=masks, non_directed=non_directed)
    else:
        return ValueError('Dataset not available')
    # if not os.path.isdir('splits'): os.makedirs('splits')
    # if data_name == 'wiki_cs':
    #     dataset = dgl.data.WikiCSDataset()
    #     g = dataset[0]
    #     try:
    #         masks = pickle.load(open(f'splits/{data_name}.splits', 'rb'))
    #         g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask'] = masks[0], masks[1], masks[2]
    #     except:
    #         train_mask, val_mask, test_mask = cross_validation_gen(g.ndata['label'])
    #         g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask'] = \
    #             train_mask[:, 0].unsqueeze(1), val_mask[:, 0].unsqueeze(1), test_mask[:, 0].unsqueeze(1)
    #         pickle.dump([g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask']],
    #                     open(f'splits/{data_name}.splits', 'wb'))
    #
    # elif data_name == 'cora_full':
    #     dataset = dgl.data.CoraFullDataset()
    #     g = dataset[0]
    #     try:
    #         masks = pickle.load(open(f'splits/{data_name}.splits', 'rb'))
    #         g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask'] = masks[0], masks[1], masks[2]
    #     except:
    #         train_mask, val_mask, test_mask = cross_validation_gen(g.ndata['label'])
    #         g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask'] = \
    #             train_mask[:, 0].unsqueeze(1), val_mask[:, 0].unsqueeze(1), test_mask[:, 0].unsqueeze(1)
    #         pickle.dump([g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask']],
    #                     open(f'splits/{data_name}.splits', 'wb'))
    #
    #
    # elif data_name == 'co_cs':
    #     dataset = dgl.data.CoauthorCSDataset()
    #     g = dataset[0]
    #     try:
    #         masks = pickle.load(open(f'splits/{data_name}.splits', 'rb'))
    #         g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask'] = masks[0], masks[1], masks[2]
    #     except:
    #         train_mask, val_mask, test_mask = cross_validation_gen(g.ndata['label'])
    #         g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask'] = \
    #             train_mask[:, 0].unsqueeze(1), val_mask[:, 0].unsqueeze(1), test_mask[:, 0].unsqueeze(1)
    #         pickle.dump([g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask']],
    #                     open(f'splits/{data_name}.splits', 'wb'))
    #
    # elif data_name == 'co_phy':
    #     dataset = dgl.data.CoauthorPhysicsDataset()
    #     g = dataset[0]
    #     try:
    #         masks = pickle.load(open(f'splits/{data_name}.splits', 'rb'))
    #         g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask'] = masks[0], masks[1], masks[2]
    #     except:
    #         train_mask, val_mask, test_mask = cross_validation_gen(g.ndata['label'])
    #         g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask'] = \
    #             train_mask[:, 0].unsqueeze(1), val_mask[:, 0].unsqueeze(1), test_mask[:, 0].unsqueeze(1)
    #         pickle.dump([g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask']],
    #                     open(f'splits/{data_name}.splits', 'wb'))
    #
    # elif data_name == 'co_photo':
    #     dataset = dgl.data.AmazonCoBuyPhotoDataset()
    #     g = dataset[0]
    #     try:
    #         masks = pickle.load(open(f'splits/{data_name}.splits', 'rb'))
    #         g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask'] = masks[0], masks[1], masks[2]
    #     except:
    #         train_mask, val_mask, test_mask = cross_validation_gen(g.ndata['label'])
    #         g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask'] = \
    #             train_mask[:, 0].unsqueeze(1), val_mask[:, 0].unsqueeze(1), test_mask[:, 0].unsqueeze(1)
    #         pickle.dump([g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask']],
    #                     open(f'splits/{data_name}.splits', 'wb'))
    #
    # elif data_name == 'co_computer':
    #     dataset = dgl.data.AmazonCoBuyComputerDataset()
    #     g = dataset[0]
    #     try:
    #         masks = pickle.load(open(f'splits/{data_name}.splits', 'rb'))
    #         g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask'] = masks[0], masks[1], masks[2]
    #     except:
    #         train_mask, val_mask, test_mask = cross_validation_gen(g.ndata['label'])
    #         g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask'] = \
    #             train_mask[:, 0].unsqueeze(1), val_mask[:, 0].unsqueeze(1), test_mask[:, 0].unsqueeze(1)
    #         pickle.dump([g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask']],
    #                     open(f'splits/{data_name}.splits', 'wb'))
    #
    # elif data_name == 'cora':
    #     dataset = dgl.data.CoraGraphDataset()
    #     g = dataset[0]
    #     if split == 'random':
    #         train_mask, val_mask, test_mask = cross_validation_gen(g.ndata['label'])
    #         g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask'] = train_mask, val_mask, test_mask
    #     else:
    #         g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask'] = \
    #             g.ndata['train_mask'].unsqueeze(1), g.ndata['val_mask'].unsqueeze(1), g.ndata['test_mask'].unsqueeze(1)
    #
    # elif data_name == 'citeseer':
    #     dataset = dgl.data.CiteseerGraphDataset()
    #     g = dataset[0]
    #     if split == 'random':
    #         train_mask, val_mask, test_mask = cross_validation_gen(g.ndata['label'])
    #         g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask'] = train_mask, val_mask, test_mask
    #     else:
    #         g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask'] = \
    #             g.ndata['train_mask'].unsqueeze(1), g.ndata['val_mask'].unsqueeze(1), g.ndata['test_mask'].unsqueeze(1)
    #
    # elif data_name == 'pubmed':
    #     dataset = dgl.data.PubmedGraphDataset()
    #     g = dataset[0]
    #     # std, mean = torch.std_mean(g.ndata['feat'], dim=0, unbiased=False)
    #     # g.ndata['feat'] = (g.ndata['feat'] - mean) / std
    #     if split == 'random':
    #         train_mask, val_mask, test_mask = cross_validation_gen(g.ndata['label'])
    #         g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask'] = train_mask, val_mask, test_mask
    #     else:
    #         g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask'] = \
    #             g.ndata['train_mask'].unsqueeze(1), g.ndata['val_mask'].unsqueeze(1), g.ndata['test_mask'].unsqueeze(1)
    #
    # elif data_name == 'actor':
    #     # dataset, _ = dgl.load_graphs(hetero_graph_path + '/actor.bin')
    #     dataset = dgl.data.ActorDataset()
    #     g = dataset[0]
    #     try:
    #         masks = pickle.load(open(f'splits/{data_name}.splits', 'rb'))
    #         g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask'] = masks[0], masks[1], masks[2]
    #     except:
    #         train_mask, val_mask, test_mask = cross_validation_gen(g.ndata['label'])
    #         g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask'] = \
    #             train_mask[:, 0].unsqueeze(1), val_mask[:, 0].unsqueeze(1), test_mask[:, 0].unsqueeze(1)
    #         pickle.dump([g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask']],
    #                     open(f'splits/{data_name}.splits', 'wb'))
    #
    # elif data_name == 'chameleon':
    #     # dataset, _ = dgl.load_graphs(hetero_graph_path + '/chameleon.bin')
    #     dataset = dgl.data.ChameleonDataset()
    #     g = dataset[0]
    #     try:
    #         masks = pickle.load(open(f'splits/{data_name}.splits', 'rb'))
    #         g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask'] = masks[0], masks[1], masks[2]
    #     except:
    #         train_mask, val_mask, test_mask = cross_validation_gen(g.ndata['label'])
    #         g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask'] = \
    #             train_mask[:, 0].unsqueeze(1), val_mask[:, 0].unsqueeze(1), test_mask[:, 0].unsqueeze(1)
    #         pickle.dump([g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask']],
    #                     open(f'splits/{data_name}.splits', 'wb'))
    #
    # elif data_name == 'squirrel':
    #     # dataset, _ = dgl.load_graphs(hetero_graph_path + '/squirrel.bin')
    #     dataset = dgl.data.SquirrelDataset()
    #     g = dataset[0]
    #     try:
    #         masks = pickle.load(open(f'splits/{data_name}.splits', 'rb'))
    #         g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask'] = masks[0], masks[1], masks[2]
    #     except:
    #         train_mask, val_mask, test_mask = cross_validation_gen(g.ndata['label'])
    #         g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask'] = \
    #             train_mask[:, 0].unsqueeze(1), val_mask[:, 0].unsqueeze(1), test_mask[:, 0].unsqueeze(1)
    #         pickle.dump([g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask']],
    #                     open(f'splits/{data_name}.splits', 'wb'))
    #
    # elif data_name == 'arxiv':
    #     from ogb.nodeproppred import DglNodePropPredDataset
    #     dataset = DglNodePropPredDataset(name='ogbn-arxiv')
    #     g = dataset[0][0]
    #     g.ndata['label'] = dataset[0][1].squeeze()
    #     splits = dataset.get_idx_split()
    #     g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask'] = \
    #         torch.zeros(g.num_nodes(), 1, dtype=torch.bool), torch.zeros(g.num_nodes(), 1,
    #                                                                      dtype=torch.bool), torch.zeros(g.num_nodes(),
    #                                                                                                     1,
    #                                                                                                     dtype=torch.bool)
    #     g.ndata['train_mask'][splits['train']] = True
    #     g.ndata['val_mask'][splits['valid']] = True
    #     g.ndata['test_mask'][splits['test']] = True
    #     # normalize graphs with discrete features
    #     from sklearn.preprocessing import StandardScaler
    #     norm = StandardScaler()
    #     norm.fit(g.ndata['feat'])
    #     g.ndata['feat'] = torch.tensor(norm.transform(g.ndata['feat'])).float()
    #
    # elif data_name == 'products':
    #     from ogb.nodeproppred import DglNodePropPredDataset
    #     dataset = DglNodePropPredDataset(name='ogbn-products')
    #     g = dataset[0][0]
    #     g.ndata['label'] = dataset[0][1].squeeze()
    #     splits = dataset.get_idx_split()
    #     g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask'] = \
    #         torch.zeros(g.num_nodes(), 1, dtype=torch.bool), torch.zeros(g.num_nodes(), 1,
    #                                                                      dtype=torch.bool), torch.zeros(g.num_nodes(),
    #                                                                                                     1,
    #                                                                                                     dtype=torch.bool)
    #     g.ndata['train_mask'][splits['train']] = True
    #     g.ndata['val_mask'][splits['valid']] = True
    #     g.ndata['test_mask'][splits['test']] = True
    #     # normalize graphs with discrete features
    #     from sklearn.preprocessing import StandardScaler, MinMaxScaler
    #     norm = StandardScaler()
    #     norm.fit(g.ndata['feat'])
    #     g.ndata['feat'] = torch.tensor(norm.transform(g.ndata['feat'])).float()
    #
    # elif data_name == 'ppi':
    #     g = []
    #     for mode in ['train', 'valid', 'test']:
    #         dataset = dgl.data.PPIDataset(mode=mode)
    #         g += ([g_ for g_ in dataset])
    #     g = dgl.batch(g)
    #     batch_num_nodes = g.batch_num_nodes()
    #     cumsum = torch.cumsum(batch_num_nodes, dim=0)
    #     train_mask, val_mask, test_mask = torch.zeros(g.num_nodes()), torch.zeros(g.num_nodes()), torch.zeros(
    #         g.num_nodes())
    #     train_mask[:cumsum[19]] = True
    #     val_mask[cumsum[19]:cumsum[21]] = True
    #     test_mask[cumsum[21]:] = True
    #     g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask'] = train_mask.unsqueeze(1), val_mask.unsqueeze(
    #         1), test_mask.unsqueeze(1)
    #     del g.ndata['_ID'], g.edata['_ID']
    #
    # else:
    #     assert Exception('Invalid Dataset')
    g = g.remove_self_loop().add_self_loop()
    if preprocess_: g = preprocess(g)
    return g


def get_elliptic():
    import torch_geometric.datasets.elliptic as elliptic
    from copy import deepcopy
    g = elliptic.EllipticBitcoinDataset('./elliptic')[0]
    train_mask, test_mask = g.train_mask, g.test_mask
    src = g.edge_index[0]
    dst = g.edge_index[1]
    split = 0.5
    feature = g.x
    label = g.y
    g = dgl.graph((src, dst))
    try:
        masks = pickle.load(open(f'./elliptic/elliptic.splits', 'rb'))
        g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask'] = masks[0], masks[1], masks[2]
    except:
        train_mask_ = deepcopy(train_mask)
        train_indices_ = train_mask.nonzero()[:, 0]
        threshold = int(len(train_indices_) * split)
        train_indices_ = train_indices_[torch.randperm(len(train_indices_))]
        train_mask[train_indices_[threshold:]] = False
        train_mask_[train_indices_[:threshold]] = False
        val_mask = train_mask_
        g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask'] = train_mask, val_mask, test_mask
        pickle.dump([g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask']],
                    open(f'./elliptic/elliptic.splits', 'wb'))
    g.ndata['feat'] = feature
    g.ndata['label'] = label

    return g

def process_data(g, masks, non_directed):
    g = g.remove_self_loop().add_self_loop()
    train_mask = torch.from_numpy(masks['train_mask'])
    val_mask = torch.from_numpy(masks['val_mask'])
    test_mask = torch.from_numpy(masks['test_mask'])
    g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask'] = train_mask, val_mask, test_mask

    print(train_mask.shape[0], val_mask.shape[0], test_mask.shape[0])
    return g