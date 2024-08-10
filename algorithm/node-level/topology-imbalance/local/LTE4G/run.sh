#!/bin/bash
#SBATCH -J test                   # 作业名为 test
#SBATCH --output=%j_mid.out    # 屏幕上的输出文件重定向到 test.out
#SBATCH -N 1                      # 作业申请 1 个节点
#SBATCH --gres=gpu:V100:1         # 单个节点使用 1 块 GPU 卡
#SBATCH -t 12:00:00
#SBATCH -x dell-gpu-23

#change
export dataset="arxiv"
export class_weight=True
export gamma=1

# non-change
export im_class_num="3"
export im_ratio="1"
export cls_og="GNN"
export rec=True
export ep_pre=200
export lr_expert="0.01"
export embedder="lte4g"
export gpu=0

python3 main.py --dataset ${dataset} --im_class_num ${im_class_num} --im_ratio ${im_ratio} --cls_og ${cls_og} --rec ${rec} --ep_pre ${ep_pre} \
               --class_weight ${class_weight} --gamma 1 --alpha 0.1 --lr_expert ${lr_expert} --embedder ${embedder} --layer gcn --gpu ${gpu} \
               --type mid --ep 60000 --expert_ep 5000 --curriculum_ep 3000 --ep_early 30000



