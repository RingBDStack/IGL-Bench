#!/bin/bash
#SBATCH -J test                   # 作业名为 test
#SBATCH --output=%j.out   # 屏幕上的输出文件重定向到 test.out
#SBATCH -N 1                      # 作业申请 1 个节点
#SBATCH --gres=gpu:V100:1         # 单个节点使用 1 块 GPU 卡
#SBATCH -t 12:00:00
#SBATCH -x dell-gpu-23

export DATASET="arxiv"

# python3 classify.py --dataset ${DATASET} --type higher

# python3 classify.py --dataset ${DATASET} --type mid

python3 classify.py --dataset ${DATASET} --type higher --seed 1

python3 classify.py --dataset ${DATASET} --type higher --seed 2

python3 classify.py --dataset ${DATASET} --type higher --seed 3

python3 classify.py --dataset ${DATASET} --type higher --seed 4

python3 classify.py --dataset ${DATASET} --type higher --seed 5