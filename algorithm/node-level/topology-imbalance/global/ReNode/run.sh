#!/bin/bash

base_params="--size-imb-type none --loss-name ce -rr 1 --model gcn"

cora_seeds=(25) 
for seed in "${cora_seeds[@]}"; do
    python transductive_run.py $base_params --shuffle-seed $seed --num-hidden 16 --lr 0.0075 --data-name cora
done

citeseer_seeds=(34)
for seed in "${citeseer_seeds[@]}"; do
    python transductive_run.py $base_params --shuffle-seed $seed --num-hidden 16 --lr 0.01 --data-name citeseer
done

chameleon_seeds=(5 )
for seed in "${chameleon_seeds[@]}"; do
    python transductive_run.py $base_params --shuffle-seed $seed --num-hidden 32 --lr 0.01 --data-name chameleon
done

squirrel_seeds=(100)
for seed in "${squirrel_seeds[@]}"; do
    python transductive_run.py $base_params --shuffle-seed $seed --num-hidden 128 --lr 0.01 --data-name squirrel
done

actor_seeds=(40)
for seed in "${actor_seeds[@]}"; do
    python transductive_run.py $base_params --shuffle-seed $seed --num-hidden 128 --lr 0.01 --data-name actor
done

pubmed_seeds=(44)
for seed in "${pubmed_seeds[@]}"; do
    python transductive_run.py $base_params --shuffle-seed $seed --num-hidden 256 --lr 0.01 --data-name pubmed
done

computers_seeds=(1)
for seed in "${computers_seeds[@]}"; do
    python transductive_run.py $base_params --shuffle-seed $seed --num-hidden 512 --lr 0.01 --data-name arxiv
done

computers_seeds=(1)
for seed in "${computers_seeds[@]}"; do
    python transductive_run.py $base_params --shuffle-seed $seed --num-hidden 256 --lr 0.01 --data-name photo
done

computers_seeds=(1)
for seed in "${computers_seeds[@]}"; do
    python transductive_run.py $base_params --shuffle-seed $seed --num-hidden 512 --lr 0.01 --data-name computers
done


