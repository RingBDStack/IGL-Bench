seeds=(40) #
topo_initials=(label)
data=ImbNode #
nlayer=2
models=(gcn) #
reweight_lr=0.01
atts=(dp)
adv_steps=(3)
reweight_weights=(0.5)
lr=0.01
dropouts=(0)
batch_size=384
n_mem=(8 16)
sup_ratios=(0.01)
inter_im_ratio=0.5
intra_im_ratios=(0.2)


for seed in ${seeds[*]}
  do
  for intra_im_ratio in ${intra_im_ratios[*]}
    do
    for topo_initial in ${topo_initials[*]}
      do
        for dropout in ${dropouts[*]}
          do
            for model in ${models[*]}
              do
                for sup_ratio in ${sup_ratios[*]}
                  do
                    for att in ${atts[*]}
                      do
                        for adv_step in ${adv_steps[*]}
                          do
                            for reweight_weight in ${reweight_weights[*]}
                              do

                                CUDA_VISIBLE_DEVICES=1 python reweight_train.py --datatype=node --dataset=$data --task=reweight --reweight_lr=$reweight_lr --model=$model --nlayer=$nlayer --epoch=2010 --lr=$lr --sup_ratio=$sup_ratio --topo_initial=$topo_initial --val_ratio=0.1 --test_ratio=0.5 --log --seed=$seed --att=$att --adv_step=$adv_step --reweight_weight=$reweight_weight --dropout=$dropout --batch_size=$batch_size --n_mem ${n_mem[*]} --intra_im_ratio=$intra_im_ratio --inter_im_ratio=$inter_im_ratio --reweight_task cls
                                
                                CUDA_VISIBLE_DEVICES=1 python reweight_train.py --datatype=node --dataset=$data --task=reweight --reweight_lr=$reweight_lr --model=$model --nlayer=$nlayer --epoch=2010 --lr=$lr --sup_ratio=$sup_ratio --topo_initial=$topo_initial --val_ratio=0.1 --test_ratio=0.5 --log --seed=$seed --att=$att --adv_step=$adv_step --reweight_weight=$reweight_weight --dropout=$dropout --batch_size=$batch_size --n_mem ${n_mem[*]} --intra_im_ratio=$intra_im_ratio --inter_im_ratio=$inter_im_ratio --reweight_task wlcls
                              done
                          done
                      done
                  done      
              done
          done
      done
    done
  done
