import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default='actor', help='dataset')
parser.add_argument("--hidden", type=int, default=32, help='hidden layer dimension')
parser.add_argument("--eta", type=float, default=0.1, help='adversarial constraint')
parser.add_argument("--mu", type=float, default=0.001, help='missing info constraint')
parser.add_argument("--lamda", type=float, default=0.0001, help='l2 parameter')
parser.add_argument("--dropout", type=float, default=0.5, help='dropout')
parser.add_argument("--k", type=int, default=5, help='num of node neighbor')
parser.add_argument("--lr", type=float, default=0.01, help='learning rate')

parser.add_argument("--arch", type=int, default=1, help='1: gcn, 2: gat, 3: graphsage')
parser.add_argument("--seed", type=int, default=0, help='Random seed')
parser.add_argument("--epochs", type=int, default=1000, help='Epochs')
parser.add_argument("--patience", type=int, default=300, help='Patience')
parser.add_argument("--id", type=int, default=0, help='gpu ids')
parser.add_argument("--g_sigma", type=float, default=1, help='G deviation')
parser.add_argument("--ablation", type=int, default=0, help='ablation mode')
parser.add_argument("--type", type=str, default='mid', help='type:lower higher mid')
args = parser.parse_args()