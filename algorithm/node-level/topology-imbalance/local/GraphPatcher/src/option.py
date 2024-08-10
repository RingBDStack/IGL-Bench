import argparse


class OptionsGNN():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize_parser()
        
    def initialize_parser(self):
        # basic parameters
        self.parser.add_argument('--dataset', type=str, default='cora', \
            choices=['wiki_cs', 'co_cs', 'co_phy', 'photo', 'computer', 'actor', \
                     'chameleon', 'squirrel', 'pubmed', 'cora', 'citeseer', 'arxiv', 'products', \
                        'cora_full', 'reddit', 'yelp', 'elliptic', 'ppi'], help='dataset')
        
        self.parser.add_argument('--hid_dim', type=int, nargs='+', default=[64])
        self.parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
        self.parser.add_argument('--device', type=int, default=0, help='index of the gpu for training')
        self.parser.add_argument('--weight_decay', type=float, default=5e-4)
        self.parser.add_argument('--norm', type=str, default='identity')
        self.parser.add_argument('--save_dir', type=str, default='temp')
        self.parser.add_argument('--wandb', action='store_true')
        self.parser.add_argument('--mp_norm', type=str, default='right')
        self.parser.add_argument('--type', type=str, default='mid')
    def parse(self):
        opt = self.parser.parse_args()
        return opt
    

class OptionsGenerator():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize_parser()
        
    def initialize_parser(self):
        # basic parameters
        self.parser.add_argument('--dataset', type=str, default='cora', \
            choices=['wiki_cs', 'co_cs', 'co_phy', 'photo', 'computer', 'actor', \
                     'chameleon', 'squirrel', 'pubmed', 'cora', 'citeseer', 'arxiv', 'products', \
                        'cora_full', 'reddit', 'yelp', 'elliptic', 'ppi'], help='dataset')
        
        self.parser.add_argument('--target_gnn', type=str, default='')
        self.parser.add_argument('--backbone', type=str, default='gcn')
        self.parser.add_argument('--hid_dim', type=int, default=1024)
        self.parser.add_argument('--warmup_steps', type=int, default=100)
        
        self.parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
        self.parser.add_argument('--device', type=int, default=0, help='index of the gpu for training')
        self.parser.add_argument('--weight_decay', type=float, default=5e-4)
        self.parser.add_argument('--degree_train', type=int, default=1)
        self.parser.add_argument('--drop_ratio', type=float, nargs='+')
        self.parser.add_argument('--three_layer', action='store_true')
        self.parser.add_argument('--k', type=int, default=3)
        self.parser.add_argument('--generation_iteration', type=int, default=-1)
        self.parser.add_argument('--total_generation_iteration', type=int, default=5)
        self.parser.add_argument('--norm', type=str, default='identity')
        self.parser.add_argument('--training_iteration', type=int, default=10000)
        self.parser.add_argument('--dropout', type=float, default=0.)
        self.parser.add_argument('--batch_size', type=int, default=128)
        self.parser.add_argument('--accumulate_step', type=int, default=1)
        self.parser.add_argument('--eval_iteration', type=int, default=100)
        self.parser.add_argument('--patience', type=int, default=30)
        self.parser.add_argument('--save_dir', type=str, default='temp')
        self.parser.add_argument('--graph_dir', type=str, default='cached_graphs')
        self.parser.add_argument('--wandb', action='store_true')
        self.parser.add_argument('--bar', action='store_true')
        self.parser.add_argument('--workers', type=int, default=10)
        self.parser.add_argument('--mp_norm', type=str, default='right')
        self.parser.add_argument('--type', type=str, default='mid')
        self.parser.add_argument('--seed', type=int, default=123)

    def parse(self):
        opt = self.parser.parse_args()
        return opt