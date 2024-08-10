GPU=$1

python3 node_generation.py \
--dataset cora \
--batch_size 64 \
--accumulate_step 1 \
--eval_iteration 10 \
--training_iteration 100000 \
--drop_ratio 0.3 0.6 0.9 \
--patience 10 \
--dropout 0.2 \
--k 2 \
--workers 24 \
--device ${GPU} 