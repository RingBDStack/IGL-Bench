#!/bin/bash
#SBATCH -J test                   # 作业名为 test
#SBATCH -o job-%j_timememory.out   # 屏幕上的输出文件重定向到 test.out
#SBATCH -N 1                      # 作业申请 1 个节点
#SBATCH --gres=gpu:V100:1         # 单个节点使用 1 块 GPU 卡
#SBATCH -t 30:00:00
#SBATCH -x dell-gpu-23

python3 node_generation.py --dataset arxiv --batch_size 16 --accumulate_step 64 --eval_iteration 10 \
                            --training_iteration 100000  --drop_ratio 0.1 0.2 0.3 0.4 0.5 --patience 10 --dropout 0.2 --k 2 \
                            --workers 8 --device 0 --type mid --seed 1


# run任务，训练模型