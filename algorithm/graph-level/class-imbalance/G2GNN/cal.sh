python main.py --dataset DD --imb_ratio 0.5 --setting knn_aug --num_train 59 \
               --num_val 59 --epochs 500 --weight_decay 0.0005 --lr 0.001
python main.py --dataset PTC_MR --imb_ratio 0.5 --setting knn_aug --num_train 59 \
               --num_val 59 --epochs 500 --weight_decay 0.0005 --lr 0.001
python main.py --dataset FRANKENSTEIN --imb_ratio 0.5 --setting knn_aug --num_train 59 \
               --num_val 59 --epochs 500 --weight_decay 0.0005 --lr 0.001
python main.py --dataset IMDB-BINARY --imb_ratio 0.5 --setting knn_aug --num_train 59 \
               --num_val 59 --epochs 500 --weight_decay 0.0005 --lr 0.001
python main.py --dataset COLLAB --imb_ratio 0.5 --setting knn_aug --num_train 59 \
               --num_val 59 --epochs 500 --weight_decay 0.0005 --lr 0.001