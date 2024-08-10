#!/bin/bash
#SBATCH -J test                   # 作业名为 test
#SBATCH --output=%j.out    # 屏幕上的输出文件重定向到 test.out
#SBATCH -N 1                      # 作业申请 1 个节点
#SBATCH --gres=gpu:V100:1         # 单个节点使用 1 块 GPU 卡
#SBATCH -t 30
#SBATCH -p dell-fast

export dataset="arxiv"

echo ${dataset}

# for SEED in {1..5};
# do 
#     python3 main.py --dataset ${dataset} --seed $SEED --type=mid
# done

python3 main.py --dataset ${dataset} --seed 4 --type=mid --epochs 10

# run任务，训练模型