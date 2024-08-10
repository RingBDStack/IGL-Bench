#!/bin/bash
#SBATCH -J test                   # 作业名为 test
#SBATCH -o job-%j.out   # 屏幕上的输出文件重定向到 test.out
#SBATCH -N 1                      # 作业申请 1 个节点
#SBATCH --gres=gpu:V100:1         # 单个节点使用 1 块 GPU 卡
#SBATCH -t 30
#SBATCH -p dell-fast

python3 train_gnn.py --dataset arxiv --mp_norm both --type higher

# run任务，训练模型