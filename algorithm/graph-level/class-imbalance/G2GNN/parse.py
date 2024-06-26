import argparse
import yaml
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bb', type=str, default='sage')
    # dataset
    parser.add_argument("--dataset", type=str, default="MUTAG",
                        help="Choose a dataset:[MUTAG, PROTEINS, DHFR, DD, NCI1, PTC-MR, REDDIT-B]")

    # model
    parser.add_argument('--n_hidden', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--prop_epochs', type=int, default=3)

    # experiments
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.005)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--imb_ratio', type=float, default=0.1)
    parser.add_argument('--num_train', type=int, default=50)
    parser.add_argument('--num_val', type=int, default=50)
    parser.add_argument('--setting', type=str, default='smote')
    # parser.add_argument("--pin_memory", type = int, default = 0)
    # parser.add_argument("--num_workers", type = int, default = 1)
    parser.add_argument('--early_stopping', type=int, default=200)

    # problem-specific
    parser.add_argument('--kernel_type', type=str, default='SP')
    parser.add_argument('--smote_k', type=int, default=10)
    parser.add_argument('--aug', type=str, default='RE')
    parser.add_argument('--drop_edge_ratio', type=float, default=0.0)
    parser.add_argument('--mask_node_ratio', type=float, default=0.0)
    parser.add_argument('--aug_num', type=int, default=2)
    parser.add_argument('--temp', type=float, default=0.5)
    parser.add_argument('--knn_layer', type=int, default=3)
    parser.add_argument('--knn_nei_num', type=int, default=3)

    return parser.parse_args()


def save_args_to_yaml(args, config_folder='config'):
    dataset_folder = os.path.join(config_folder, args.dataset)
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    filename = os.path.join(dataset_folder, f"{args.bb}.yml")

    with open(filename, 'w') as file:
        yaml.dump(args.__dict__, file, default_flow_style=False)