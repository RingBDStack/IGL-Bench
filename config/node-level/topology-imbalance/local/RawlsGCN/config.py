import argparse

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