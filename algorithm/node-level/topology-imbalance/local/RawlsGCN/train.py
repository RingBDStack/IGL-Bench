import argparse
import numpy as np
import torch
import time

from models.rawlsgcn_graph import RawlsGCNGraph
from models.rawlsgcn_grad import RawlsGCNGrad
from utils import data_loader, utils
from utils.trainer_rawlsgcn_graph import PreProcessingTrainer
from utils.trainer_rawlsgcn_grad import InProcessingTrainer


# parse argument
parser = argparse.ArgumentParser()
parser.add_argument(
    "--enable_cuda", action="store_true", default=True, help="Enable CUDA training."
)
parser.add_argument(
    "--device_number", type=int, default=0, help="Disables CUDA training."
)
parser.add_argument(
    "--dataset", type=str, default="amazon_photo", help="Dataset to train."
)
parser.add_argument(
    "--model", type=str, default="rawlsgcn_graph", help="Model to train."
)
parser.add_argument(
    "--split_seed",
    type=int,
    default=1684992425,
    help="Random seed to generate data splits.",
)
parser.add_argument(
    "--num_epoch", type=int, default=100, help="Number of epochs to train."
)
parser.add_argument("--lr", type=float, default=0.05, help="Initial learning rate.")
parser.add_argument(
    "--weight_decay",
    type=float,
    default=5e-4,
    help="Weight decay (L2 loss on parameters).",
)
parser.add_argument("--hidden", type=int, default=64, help="Number of hidden units.")
parser.add_argument(
    "--dropout", type=float, default=0.5, help="Dropout rate (1 - keep probability)."
)
parser.add_argument(
    "--loss",
    type=str,
    default="negative_log_likelihood",
    help="Loss function (negative_log_likelihood or cross_entropy).",
)
parser.add_argument(
    "--type",
    type=str,
    default="mid",
    help="dataset type",
)

args = parser.parse_args()
args.cuda = args.enable_cuda and torch.cuda.is_available()
if args.cuda:
    device = torch.device(f"cuda:{args.device_number}")
else:
    device = torch.device("cpu")

def generate_split(dataset, t):
    # # set random seed
    # if args.split_seed is not None:
    #     np.random.seed(args.split_seed)
    #     torch.manual_seed(args.split_seed)
    #     if args.cuda:
    #         torch.cuda.manual_seed(args.split_seed)
    #
    # # generate splits
    # return utils.random_split(dataset)
    masks = torch.load(f'../../data_mask/{dataset.dataset_name}_{t}_.pth')
    node_id = torch.arange(0, dataset.num_nodes)

    idx_train = node_id[masks['train_mask']]
    idx_val = node_id[masks['val_mask']]
    idx_test = node_id[masks['test_mask']]

    print(idx_train.shape[0], idx_val.shape[0], idx_test.shape[0])

    return {
        "train_idx": idx_train,
        "val_idx": idx_val,
        "test_idx": idx_test,
    }


def run_exp(dataset, split, configs):
    # set splits
    dataset.set_random_split(split)

    # set random seed
    np.random.seed(configs["seed"])
    torch.manual_seed(configs["seed"])
    if args.cuda:
        torch.cuda.manual_seed(configs["seed"])

    # init model
    if args.model == "rawlsgcn_graph":
        model = RawlsGCNGraph(
            nfeat=dataset.num_node_features,
            nhid=args.hidden,
            nclass=dataset.num_classes,
            dropout=args.dropout,
        )
    elif args.model == "rawlsgcn_grad":
        model = RawlsGCNGrad(
            nfeat=dataset.num_node_features,
            nhid=args.hidden,
            nclass=dataset.num_classes,
            dropout=args.dropout,
        )
    else:
        raise ValueError("Invalid model name!")

    # train and test
    if args.model == "rawlsgcn_graph":
        trainer = PreProcessingTrainer(
            configs=configs, data=dataset, model=model, on_gpu=args.cuda, device=device
        )
    elif args.model == "rawlsgcn_grad":
        trainer = InProcessingTrainer(
            configs=configs, data=dataset, model=model, on_gpu=args.cuda, device=device
        )
    else:
        raise ValueError("Invalid model name!")

    #TODO metrics
    trainer.train()
    trainer.test()


if __name__ == "__main__":
    # update configs
    dataset_configs = {
        "name": args.dataset,
        "is_ratio": False,
        "split_by_class": True,
        "num_train": 20,
        "num_val": 500,
        "num_test": 1000,
        "ratio_train": 0.8,
        "ratio_val": 0.1,
    }

    #TODO load data
    dataset = data_loader.GraphDataset(dataset_configs)

    #TODO return train/val/test get random splits
    split = generate_split(dataset, args.type)

    # train
    if args.model == "rawlsgcn_grad":
        dataset.preprocess(type="laplacian")
    elif args.model == "rawlsgcn_graph":
        dataset.preprocess(type="doubly_stochastic_laplacian")
    else:
        raise ValueError("Invalid model name!")

    random_seed_list = list(range(5))

    for random_seed in random_seed_list:
        # update configs
        configs = {
            "name": args.dataset,
            "model": args.model,
            "num_epoch": args.num_epoch,
            "hidden": args.hidden,
            "weight_decay": args.weight_decay,
            "type": args.type,
            "lr": args.lr,
            "loss": args.loss,
            "seed": random_seed,
            "save_path": f"ckpts/{args.dataset}/{args.model}/lr={args.lr}_nepochs={args.num_epoch}_decay={args.weight_decay}_seed={random_seed}.pt",
        }

        torch.cuda.reset_max_memory_allocated()
        start_time = time.time()

        run_exp(dataset, split, configs)

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time:.4f} seconds")

        max_memory = torch.cuda.max_memory_allocated()
        print(f"Peak memory usage: {max_memory / (1024**2):.4f} MB")
