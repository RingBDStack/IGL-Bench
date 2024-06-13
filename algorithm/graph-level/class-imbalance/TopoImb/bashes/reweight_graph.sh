seeds=(24 4) # 24 40 42)
topo_initials=(label)
datasets=(ImbTopo) #ImbTopo
nlayer=2
model=sage #
reweight_lr=0.01
atts=(dp)
adv_steps=(3) 
reweight_weights=(0.5)
lrs=(0.01)
dropouts=(0)
batch_size=384
sup_ratios=(0.05)
n_mem=(8 16)
reweighters=(structATT)

for seed in ${seeds[*]}
  do
  for data in ${datasets[*]}
    do
    for topo_initial in ${topo_initials[*]}
      do
        for dropout in ${dropouts[*]}
          do
            for lr in ${lrs[*]}
              do
                for sup_ratio in ${sup_ratios[*]}
                  do
                    for att in ${atts[*]}
                      do
                        for adv_step in ${adv_steps[*]}
                          do
                            for reweight_weight in ${reweight_weights[*]}
                              do
                                for reweighter in ${reweighters[*]}
                                  do
                                    #CUDA_VISIBLE_DEVICES=3 python reweight_train.py --datatype=graph --dataset=$data --task=reweight --reweight_lr=$reweight_lr --model=$model --nlayer=$nlayer --epoch=2010 --lr=$lr --sup_ratio=$sup_ratio --topo_initial=$topo_initial --val_ratio=0.1 --test_ratio=0.5 --log --seed=$seed --att=$att --adv_step=$adv_step --reweight_weight=$reweight_weight --dropout=$dropout --batch_size=$batch_size --n_mem ${n_mem[*]} --reweighter=$reweighter
                                    
                                    #CUDA_VISIBLE_DEVICES=3 python reweight_train.py --datatype=graph --dataset=$data --task=reweight --reweight_lr=$reweight_lr --model=$model --nlayer=$nlayer --epoch=2010 --lr=$lr --sup_ratio=$sup_ratio --topo_initial=$topo_initial --val_ratio=0.1 --test_ratio=0.5 --log --seed=$seed --att=$att --adv_step=$adv_step --reweight_weight=$reweight_weight --dropout=$dropout --batch_size=$batch_size --n_mem ${n_mem[*]} --reweight_task cls --reweighter=$reweighter

                                    CUDA_VISIBLE_DEVICES=3 python reweight_train.py --datatype=graph --dataset=$data --task=reweight --reweight_lr=$reweight_lr --model=$model --nlayer=$nlayer --epoch=2010 --lr=$lr --sup_ratio=$sup_ratio --topo_initial=$topo_initial --val_ratio=0.1 --test_ratio=0.5 --log --seed=$seed --att=$att --adv_step=$adv_step --reweight_weight=$reweight_weight --dropout=$dropout --batch_size=$batch_size --n_mem ${n_mem[*]} --reweight_task wlcls --reweighter=$reweighter
                                  done
                              done
                          done
                      done
                  done      
              done
          done
      done
    done
  done

#CUDA_VISIBLE_DEVICES=3 python reweight_train.py --datatype=graph --dataset=$data --task=reweight --reweight_lr=$reweight_lr --model=$model --nlayer=$nlayer --epoch=2010 --lr=$lr --sup_ratio=$sup_ratio --topo_initial=$topo_initial --val_ratio=0.1 --test_ratio=0.5 --log --seed=$seed --att=$att --adv_step=$adv_step --reweight_weight=$reweight_weight --dropout=$dropout --batch_size=$batch_size --n_mem ${n_mem[*]} --reweighter=$reweighter
