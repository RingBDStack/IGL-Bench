#!/bin/bash
#SBATCH -J test                   # 作业名为 test
#SBATCH --output=%j_timememory.out    # 屏幕上的输出文件重定向到 test.out
#SBATCH -N 1                      # 作业申请 1 个节点
#SBATCH --gres=gpu:V100:1         # 单个节点使用 1 块 GPU 卡
#SBATCH -t 12:00:00
#SBATCH -x dell-gpu-23

for SEED in {1..5};
do 
    python3 main.py --dataset=actor --eta=0.1 --mu=0.001 --k=5 --seed $SEED --type=mid
done


# run任务，训练模型