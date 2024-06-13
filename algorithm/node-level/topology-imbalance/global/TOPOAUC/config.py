import os,sys
import argparse
import yaml

dirs={
    "LOGDIR":"./logs/",
    "DATA_PATH":"./datasets/"
}


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--valid_each', default=30, type=int, help="the validation size of each class")
    parser.add_argument('--imb_ratio',  default=5.0, type=float, help="the ratio of the majority class size to the minoriry class size",choices=[5.0,10.0,15.0,20.0,30.0])
    parser.add_argument('--labeling_ratio', default=0.1, type=float, help="the labeling ratio of the dataset")
    parser.add_argument('--is_imb',default=False,type=bool,help="is or not imbalance type")
    parser.add_argument('--pagerank-prob', default=0.15, type=float,help="probility of going down instead of going back to the starting position in the random walk")
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--rn-base-weight',  '-rbw',  default=0.5, type=float, help="the base  weight of renode reweight")
    parser.add_argument('--rn-scale-weight', '-rsw',  default=1.0, type=float, help="the scale weight of renode reweight")
    parser.add_argument('--dataset', default="cora", type=str,help="datasets")
    parser.add_argument('--run-split-num', default=1, type=int, help='run N different split times')
    parser.add_argument('--run-init-num',  default=10, type=int, help='run N different init seeds')

    
    
    parser.add_argument('--balance_train_each', default=20, type=int, help="the training size of each class, used in none imbe type")
    
    parser.add_argument('--model', default='gcn', type=str) #gcn gat ppnp sage cheb sgc
    parser.add_argument('--num-hidden', default=16, type=int)
    parser.add_argument('--num-feature', default=745, type=int)
    parser.add_argument('--num-class', default=7, type=int)
    parser.add_argument('--num-layer', default=2, type=int)
    parser.add_argument('--saved-model', default='best-model.pt', type=str)

    parser.add_argument("--loss",default="ExpGAUC",choices=["ce","focal","re-weight","cb-softmax","ExpGAUC","HingeGAUC","SqGAUC"])
    parser.add_argument('--pair_ner_diff',   default=1,   type=int,   help="add our topology weight")
    
    parser.add_argument('--factor_focal', default=2.0,    type=float, help="alpha in Focal Loss")
    parser.add_argument('--factor_cb',    default=0.9999, type=float, help="beta  in CB Loss")
    #ReNode
    parser.add_argument('--renode-reweight', '-rr',   default=0,   type=int,   help="switch of ReNode") # 0 (not use) or 1 (use)
    

    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--lr-decay-epoch', default=10, type=int)
    parser.add_argument('--lr-decay-rate', default=0.95, type=float)
    parser.add_argument('--weight-decay', default=0, type=float)
    parser.add_argument('--dropout', default=0.7, type=float)
    parser.add_argument('--epoch', default=500, type=int)
    parser.add_argument('--least-epoch', default=40, type=int)
    parser.add_argument('--early-stop', default=20, type=int,choices=[-1,5,20,30])

    
    parser.add_argument('--ppr-topk', default=-1,type=int)

    parser.add_argument('--nni', default=0,type=int)

    parser.add_argument('--weight_sub_dim', default=64,type=int)
    parser.add_argument('--weight_inter_dim', default=64,type=int)
    parser.add_argument('--weight_global_dim', default=64,type=int)
    parser.add_argument('--topo_dim', default=64,type=int)
    parser.add_argument('--beta', default=0.5,type=float)
    parser.add_argument('--gamma', default=1.0,type=int)

    parser.add_argument('--warm_up_epoch',  default=0, type=int, help='')
    parser.add_argument('--warm_up_loss',  default="focal",type=str,choices=["ce","focal","re-weight","cb-softmax"])
    
    parser.add_argument('--save_repeated_res',  default=1, type=int, help='')
    parser.add_argument('--shuffle_seed', default=80, type=int, help="shuffle")
    
    opts = parser.parse_args()
    opts.shuffle_seed_list = [i for i in range(opts.run_split_num)]
    opts.seed_list         = [i for i in range(opts.run_init_num) ]

    opts.gem_file = "{}/{}_{:.2f}_gem.pt".format(dirs["DATA_PATH"],opts.dataset,opts.pagerank_prob) # the pre-computed global effect matrix
    opts.saved_model="saved_models/{}_{}_{}_is_renode_{}_best_model.pt".format(opts.dataset,opts.model,opts.loss,opts.renode_reweight)
    save_to_yaml(opts)
    return opts

def save_to_yaml(opt):
    filename = f"../config/{opt.dataset}.yml"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w') as file:
        yaml.dump(opt, file, default_flow_style=False)