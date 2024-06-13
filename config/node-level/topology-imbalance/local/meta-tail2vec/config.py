import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--type", type=str, default='mid', help='type:lower higher mid')
parser.add_argument("--dataset_name", type=str, default='photo', help='dataset name')
parser.add_argument("--seed", type=int, default=1, help='dataset name')
args = parser.parse_args()
