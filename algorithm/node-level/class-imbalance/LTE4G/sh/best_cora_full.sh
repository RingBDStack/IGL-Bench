#!/bin/bash
export dataset="cora_full"
export im_class_num="3"
export im_ratio="1"
export cls_og="MLP"
export rec=False
export ep_pre=50
export class_weight=True
export gamma=1
export alpha=0.6
export lr_expert="0.01"
export embedder="lte4g"
export pareto="pareto_73"
export gpu=3

python main.py --dataset ${dataset} --im_class_num ${im_class_num} --im_ratio ${im_ratio} --cls_og ${cls_og} --sep_class ${pareto} --rec ${rec} --ep_pre ${ep_pre} \
               --class_weight ${class_weight} --gamma ${gamma} --alpha ${alpha} --lr_expert ${lr_expert} --embedder ${embedder} --layer gcn --gpu ${gpu}


