#!/bin/bash
#SBATCH -J test                   # 作业名为 test
#SBATCH --output=%jtimememory.out    # 屏幕上的输出文件重定向到 test.out
#SBATCH -N 1                      # 作业申请 1 个节点
#SBATCH --gres=gpu:V100:1         # 单个节点使用 1 块 GPU 卡
#SBATCH -t 30
#SBATCH -p dell-fast

# for LR in 0.075 0.05 0.025 0.01 0.0075 0.005 0.0025;
# for LR in 0.05 0.025 0.01;
# do
#     python3 train.py --model rawlsgcn_graph --dataset=arxiv --lr ${LR} --num_epoch 2000 --type=higher
# done

python3 train.py --model rawlsgcn_graph --dataset actor --lr 0.005 --type=mid

# run任务，训练模型