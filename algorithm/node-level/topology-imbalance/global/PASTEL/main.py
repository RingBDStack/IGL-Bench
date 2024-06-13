import argparse
import yaml
import torch
import numpy as np
from collections import defaultdict, OrderedDict

from src.model_handler import ModelHandler


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_config(config_path="config.yml"):
    with open(config_path, "r") as setting:
        config = yaml.load(setting, Loader=yaml.FullLoader)
    return config


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', '--config', required=True, type=str, help='path to the config file')
    parser.add_argument('--multi_run', action='store_true', help='whether open multiple run')
    parser.add_argument('--seed', default=80,type=int, help='seed')
    parser.add_argument('--rseed', default=1,type=int, help='random_seed')
    args = vars(parser.parse_args())
    return args


def print_config(config):
    print("******************** MODEL CONFIGURATION ********************")
    for key in sorted(config.keys()):
        val = config[key]
        keystr = "{}".format(key) + (" " * (24 - len(key)))
        print("{} ==> {}".format(keystr, val))
    print("********************* END CONFIGURATION *********************")


def grid(kwargs):
    class MncDc:

        def __init__(self, a):
            self.a = a
        def __call__(self):
            return self.a

    def merge_dicts(*dicts):
        from functools import reduce

        def merge_two_dicts(x, y):
            z = x.copy()
            z.update(y)
            return z
        return reduce(lambda a, nd: merge_two_dicts(a, nd if nd else {}), dicts, {})

    sin = OrderedDict({k: v for k, v in kwargs.items() if isinstance(v, list)})
    for k, v in sin.items():
        copy_v = []
        for e in v:
            copy_v.append(MncDc(e) if isinstance(e, tuple) else e)
        sin[k] = copy_v

    grd = np.array(np.meshgrid(*sin.values()), dtype=object).T.reshape(-1, len(sin.values()))
    return [merge_dicts(
        {k: v for k, v in kwargs.items() if not isinstance(v, list)},
        {k: vv[i]() if isinstance(vv[i], MncDc) else vv[i] for i, k in enumerate(sin)}
    ) for vv in grd]



def main(config):
    print_config(config)
    set_random_seed(config['random_seed'])
    model = ModelHandler(config)
    model.train()
    model.test()


if __name__ == '__main__':
    args = get_args()
    config = get_config(args['config'])
    config['seed'] = args['seed']
    config['random_seed'] = args['rseed']
    main(config)
