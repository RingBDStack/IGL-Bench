from __future__ import division
from __future__ import print_function
from time import strftime, localtime
import tensorflow as tf
import argparse
import numpy as np
import time
import torch
import GPUtil
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score

from util import load_data
from models import BaseModel
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def parse_args():
    parser = argparse.ArgumentParser(description="Run the DEMO-Net.")
    parser.add_argument('--dataset', nargs='?', default='cora',
                        help='Choose a dataset: brazil, europe or usa')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs.')
    parser.add_argument('--dropout', type=int, default=0.1,
                        help='dropout rate (1 - keep probability).')
    parser.add_argument('--patience', type=int, default=100,
                        help='patience to update the parameters.')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='weight for l2 loss on embedding matrix')
    parser.add_argument('--hash_dim', type=int, default=256,
                        help='Feature hashing dimension')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden units')
    parser.add_argument('--n_hash_kernel', type=int, default=1,
                        help='Number of hash kernels')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='Number of hidden layers')
    parser.add_argument('--type', type=str, default='mid',
                        help='type')
    parser.add_argument('--seed', type=int, default=111,
                        help='random seed')
    return parser.parse_args()


def construct_placeholder(num_nodes, fea_size, num_classes):
    with tf.name_scope('input'):
        placeholders = {
            'labels': tf.compat.v1.placeholder(tf.float32, shape=(None, num_classes), name='labels'),
            'features': tf.compat.v1.placeholder(tf.float32, shape=(num_nodes, fea_size), name='features'),
            'dropout': tf.compat.v1.placeholder_with_default(0., shape=(), name='dropout'),
            'masks': tf.compat.v1.placeholder(dtype=tf.int32, shape=(num_nodes,), name='masks'),
        }
        return placeholders


def train(args, data):

    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, degreeTasks, neighbor_list = data
    features = features.todense()
    num_nodes, fea_size = features.shape
    num_classes = y_train.shape[1]

    placeholders = construct_placeholder(num_nodes, fea_size, num_classes)

    model = BaseModel(placeholders, degreeTasks, neighbor_list, num_classes, fea_size, hash_dim=args.hash_dim,
                      hidden_dim=args.hidden_dim, num_hash=args.n_hash_kernel, num_layers=args.n_layers)

    logits = model.inference()
    log_resh = tf.reshape(logits, [-1, num_classes])
    lab_resh = tf.reshape(placeholders['labels'], [-1, num_classes])
    msk_resh = tf.reshape(placeholders['masks'], [-1])
    loss = model.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
    accuracy = model.masked_accuracy(log_resh, lab_resh, msk_resh)
    find_logits = model.find_logits(log_resh, lab_resh, msk_resh)
    find_label = model.find_label(log_resh, lab_resh, msk_resh)
    find_mask = model.find_mask(log_resh, lab_resh, msk_resh)

    train_op = model.training(loss, lr=args.lr, l2_coef=args.weight_decay)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer())

    vloss_min = np.inf
    vacc_max = 0.0
    curr_step = 0

    with tf.Session() as sess:
        sess.run(init_op)
        vacc_early_model = 0.0
        vlss_early_model = 0.0

        for epoch in range(args.epochs):
            train_feed_dict = {}
            train_feed_dict.update({placeholders['labels']: y_train})
            train_feed_dict.update({placeholders['features']: features})
            train_feed_dict.update({placeholders['dropout']: args.dropout})
            train_feed_dict.update({placeholders['masks']: train_mask})
            _, loss_value_tr, acc_tr, temp_logits, temp_labels, temp_mask = sess.run([train_op, loss, accuracy, find_logits, find_label, find_mask], feed_dict=train_feed_dict)

            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                print(f"total{gpu.memoryTotal} used memory {gpu.memoryUsed}MB free{gpu.memoryFree}")
            val_feed_dict = {}
            val_feed_dict.update({placeholders['labels']: y_val})
            val_feed_dict.update({placeholders['features']: features})
            val_feed_dict.update({placeholders['dropout']: 0.0})
            val_feed_dict.update({placeholders['masks']: val_mask})
            loss_value_val, acc_val, temp_logits, temp_labels, temp_mask = sess.run([loss, accuracy, find_logits, find_label, find_mask], feed_dict=val_feed_dict)

            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                print(f"total{gpu.memoryTotal} used memory {gpu.memoryUsed}MB free{gpu.memoryFree}")
            temp_logits = temp_logits[val_mask]
            temp_labels = temp_labels[val_mask]
            indices = np.argmax(temp_logits, axis=1)
            one_label = np.argmax(temp_labels, axis=1)
            f1 = f1_score(one_label, indices, average='macro')
            bacc = balanced_accuracy_score(one_label, indices)
            roc_auc = roc_auc_score(temp_labels, temp_logits, multi_class="ovo")

            print('Training epoch %d-th: loss = %.5f, acc = %.5f | Val: loss = %.5f, acc = %.5f bacc = %.5f f1 = %.5f roc_auc = %.5f' %
                  (epoch + 1, loss_value_tr, acc_tr, loss_value_val, acc_val, bacc, f1, roc_auc))

            if acc_val >= vacc_max or loss_value_val <= vloss_min:
                if acc_val >= vacc_max and loss_value_val <= vloss_min:
                    vacc_early_model = acc_val
                    vlss_early_model = loss_value_val
                vacc_max = np.max((acc_val, vacc_max))
                vloss_min = np.min((loss_value_val, vloss_min))
                curr_step = 0

                test_feed_dict = {}
                test_feed_dict.update({placeholders['labels']: y_test})
                test_feed_dict.update({placeholders['features']: features})
                test_feed_dict.update({placeholders['dropout']: 0.0})
                test_feed_dict.update({placeholders['masks']: test_mask})
                best_loss_value_test, best_acc_test, temp_logits, temp_labels, temp_mask = sess.run(
                    [loss, accuracy, find_logits, find_label, find_mask], feed_dict=test_feed_dict)

                temp_logits = temp_logits[test_mask]
                temp_labels = temp_labels[test_mask]
                # folder_name = "embedding"
                # if not os.path.exists(folder_name):
                #     os.makedirs(folder_name)

                # file_name = os.path.join(folder_name, f"{args.dataset}_{args.seed}.pt")
                # torch.save(temp_logits, file_name)
                indices = np.argmax(temp_logits, axis=1)
                one_label = np.argmax(temp_labels, axis=1)
                best_f1 = f1_score(one_label, indices, average='macro')
                best_bacc = balanced_accuracy_score(one_label, indices)
                best_roc_auc = roc_auc_score(temp_labels, temp_logits, multi_class="ovo")

            else:
                curr_step += 1
                if curr_step == args.patience:
                    print('Early stop! Min loss: ', vloss_min, ', Max accuracy: ', vacc_max)
                    print('Early stop model validation loss: ', vlss_early_model, ', accuracy: ', vacc_early_model)
                    break

        print('Test result; loss:', best_loss_value_test, '; accuracy:', best_acc_test, '; f1:', best_f1, '; bacc:', best_bacc,
              '; roc_auc:', best_roc_auc)

        sess.close()


if __name__ == '__main__':
    time_stamp = strftime('%Y_%m_%d_%H_%M_%S', localtime())
    print("The time of running the codes: ", time_stamp)
    args = parse_args()

    # Set random seed
    seed = args.seed
    np.random.seed(args.seed)
    tf.compat.v1.set_random_seed(args.seed)

    data = load_data(args.dataset, args.type)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # 异常处理
            print(e)

    start_time = time.time()
    train(args, data)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.4f} seconds")
