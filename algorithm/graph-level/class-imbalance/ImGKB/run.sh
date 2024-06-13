python train.py --dataset MCF-7 --num-layers 2 --max-step 4 --hidden-graphs 6 --size-hidden-graphs 4 --hidden-dim 96  --beta 1 --n_split 2
python train.py --dataset SW-620 --num-layers 2 --max-step 2 --hidden-graphs 3 --size-hidden-graphs 8 --hidden-dim 96  --beta 0.2 --n_split 10
python train.py --dataset PC-3 --num-layers 2 --max-step 4 --hidden-graphs 8 --size-hidden-graphs 8 --hidden-dim 96  --beta 0.1 --n_split 10
python train.py --dataset MOLT-4 --num-layers 2 --max-step 4 --hidden-graphs 4 --size-hidden-graphs 6 --hidden-dim 96  --beta 0.8 --n_split 10

