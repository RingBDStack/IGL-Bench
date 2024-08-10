#!/bin/bash
#SBATCH -J test                   # 作业名为 test
#SBATCH -o job-%j.out   # 屏幕上的输出文件重定向到 test.out
#SBATCH -N 1                      # 作业申请 1 个节点
#SBATCH --gres=gpu:V100:1         # 单个节点使用 1 块 GPU 卡
#SBATCH -t 30
#SBATCH -p dell-fast

export dataset="cora"

# python3 main.py --dataset ${dataset}  --train_which=TeacherGNN --exp_mode coldbrew \
#                     --whetherHasSE=100 --se_reg=32 --want_headtail=1 --num_layers=2 --use_special_split=1 \
#                     --type lower --N_exp 5

# python3 main.py --dataset ${dataset}  --train_which=TeacherGNN --exp_mode coldbrew \
#                     --whetherHasSE=100 --se_reg=32 --want_headtail=1 --num_layers=2 --use_special_split=1 \
#                     --type mid --N_exp 5

python3 main.py --dataset ${dataset}  --train_which=TeacherGNN --exp_mode coldbrew \
                    --whetherHasSE=100 --se_reg=32 --want_headtail=1 --num_layers=2 --use_special_split=1 \
                    --type mid --N_exp 5
