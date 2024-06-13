import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run the DEMO-Net.")
    parser.add_argument('--dataset', nargs='?', default='cora',
                        help='Choose a dataset: brazil, europe or usa')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs.')
    parser.add_argument('--dropout', type=int, default=0.1,
                        help='dropout rate (1 - keep probability).')
    parser.add_argument('--patience', type=int, default=100,
                        help='patience to update the parameters.')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='weight for l2 loss on embedding matrix')
    parser.add_argument('--hash_dim', type=int, default=256,
                        help='Feature hashing dimension')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden units')
    parser.add_argument('--n_hash_kernel', type=int, default=1,
                        help='Number of hash kernels')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='Number of hidden layers')
    parser.add_argument('--type', type=str, default='mid',
                        help='type')
    parser.add_argument('--seed', type=int, default=111,
                        help='random seed')
    return parser.parse_args()
