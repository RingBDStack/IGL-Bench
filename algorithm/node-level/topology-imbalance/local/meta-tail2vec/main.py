import os
import numpy as np
import argparse
import tensorflow as tf
import time
import GPUtil

from data_generator import DataGenerator
from maml import MAML

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--test', action='store_true', default=False, help='set for test, otherwise train')
parser.add_argument("--type", type=str, default='mid', help='type:lower higher mid')
parser.add_argument("--dataset_name", type=str, default='cora', help='dataset name')
parser.add_argument("--seed", type=int, default=123, help='random seed')
parser.add_argument("--epoch", type=int, default=2000, help='random seed')
args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train(model, sess, batch_num, epoch):
    prelosses, postlosses = [], []

    for iteration in range(epoch):
        ops = [model.meta_op]

        if iteration % 20 == 0:
            ops.extend([model.summ_op, model.query_losses[0], model.query_losses[-1]])

        result = sess.run(ops)
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            print(f"total{gpu.memoryTotal} used memory {gpu.memoryUsed}MB free{gpu.memoryFree}")
        if iteration % 20 == 0:
            prelosses.append(result[2])
            postlosses.append(result[3])
            print(iteration, '\tloss:', np.mean(prelosses), '=>', np.mean(postlosses))
            prelosses, postlosses = [], []


def test(model, sess, dataset_name, test_n, t):
    test_preds = []
    fw_1 = open('./data/' + dataset_name + f'/{t}_result_1.csv', 'w')
    fw_2 = open('./data/' + dataset_name + f'/{t}_result_2.csv', 'w')
    fw_3 = open('./data/' + dataset_name + f'/{t}_result_3.csv', 'w')
    fw_4 = open('./data/' + dataset_name + f'/{t}_result_4.csv', 'w')
    fw_5 = open('./data/' + dataset_name + f'/{t}_result_5.csv', 'w')
    for i in range(test_n):
        if i % 100 == 1:
            print(i)
        ops = [model.test_query_preds, model.query_nodes]
        result, nodes = sess.run(ops)
        for n in range(4):
            fw_1.write(str(int(nodes[0][n][0])) + ' ')
            temp = [str(x) for x in result[0][n].tolist()]
            fw_1.write(' '.join(temp))
            fw_1.write('\n')
        for n in range(4):
            fw_2.write(str(int(nodes[1][n][0])) + ' ')
            temp = [str(x) for x in result[1][n].tolist()]
            fw_2.write(' '.join(temp))
            fw_2.write('\n')
        for n in range(4):
            fw_3.write(str(int(nodes[2][n][0])) + ' ')
            temp = [str(x) for x in result[2][n].tolist()]
            fw_3.write(' '.join(temp))
            fw_3.write('\n')
        for n in range(4):
            fw_4.write(str(int(nodes[3][n][0])) + ' ')
            temp = [str(x) for x in result[3][n].tolist()]
            fw_4.write(' '.join(temp))
            fw_4.write('\n')
        for n in range(4):
            fw_5.write(str(int(nodes[4][n][0])) + ' ')
            temp = [str(x) for x in result[4][n].tolist()]
            fw_5.write(' '.join(temp))
            fw_5.write('\n')
    fw_1.close()
    fw_2.close()
    fw_3.close()
    fw_4.close()
    fw_5.close()
    print('Done.')


def main():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # 异常处理
            print(e)
    training = not args.test
    main_dir = './data/'
    dataset_name = args.dataset_name
    t = args.type
    kshot = 5
    meta_batchsz = 4
    k = 5
    batch_num =50000
    emb_dim = 1433
    test_num = 100
    if dataset_name == 'flickr':
        batch_num = 15
    elif dataset_name == 'wiki':
        batch_num = 10000
    elif dataset_name == 'email':
        batch_num = 5000
    elif dataset_name == 'cora':
        batch_num = 273
        test_num = 541
        emb_dim = 1433
    elif dataset_name == 'citeseer':
        batch_num = 330
        test_num = 667
        emb_dim = 3703
    elif dataset_name == 'pubmed':
        batch_num = 1971
        test_num = 3944
        emb_dim = 500
    elif dataset_name == 'squirrel':
        batch_num = 520
        test_num = 1041
        emb_dim = 2089
    elif dataset_name == 'actor':
        batch_num = 760
        test_num = 1521
        emb_dim = 931
    elif dataset_name == 'chameleon':
        batch_num = 230
        test_num = 455
        emb_dim = 2325
    elif dataset_name == 'photo':
        batch_num = 768
        test_num = 1529
        emb_dim = 745
    elif dataset_name == 'computer':
        batch_num = 1380
        test_num = 2753
        emb_dim = 767
    elif dataset_name == 'arxiv':
        emb_dim = 128
        if t == 'mid':
            batch_num = 15752
            test_num = 34461
        elif t == 'lower':
            batch_num = 16943
            test_num = 33867
        elif t == 'higher':
            batch_num = 16945
            test_num = 33867
        
    else:
        batch_num = 273
    db = DataGenerator(main_dir, dataset_name, kshot, meta_batchsz, t, batch_num)
    if training:
        node_tensor, label_tensor, data_tensor = db.make_data_tensor(training=True)
        support_n = tf.slice(node_tensor, [0, 0, 0], [-1, kshot, -1], name='support_n')
        query_n = tf.slice(node_tensor, [0, kshot, 0], [-1, -1, -1], name='query_n')
        support_x = tf.slice(data_tensor, [0, 0, 0], [-1, kshot, -1], name='support_x')
        query_x = tf.slice(data_tensor, [0, kshot, 0], [-1, -1, -1], name='query_x')
        support_y = tf.slice(label_tensor, [0, 0, 0], [-1, kshot, -1], name='support_y')
        query_y = tf.slice(label_tensor, [0, kshot, 0], [-1, -1, -1], name='query_y')

    node_tensor, label_tensor, data_tensor = db.make_data_tensor(training=False)
    support_n_test = tf.slice(node_tensor, [0, 0, 0], [-1, kshot, -1], name='support_n_test')
    query_n_test = tf.slice(node_tensor, [0, kshot, 0], [-1, -1, -1], name='query_n_test')
    support_x_test = tf.slice(data_tensor, [0, 0, 0], [-1, kshot, -1], name='support_x_test')
    query_x_test = tf.slice(data_tensor, [0, kshot, 0], [-1, -1, -1], name='query_x_test')
    support_y_test = tf.slice(label_tensor, [0, 0, 0], [-1, kshot, -1], name='support_y_test')
    query_y_test = tf.slice(label_tensor, [0, kshot, 0], [-1, -1, -1], name='query_y_test')

    model = MAML(emb_dim)

    model.build(support_n, support_x, support_y, query_n, query_x, query_y, k, meta_batchsz, mode='train')
    model.build(support_n_test, support_x_test, support_y_test, query_n_test, query_x_test, query_y_test, k,
                meta_batchsz, mode='test')
    model.summ_op = tf.summary.merge_all()

    all_vars = filter(lambda x: 'meta_optim' not in x.name, tf.trainable_variables())
    for p in all_vars:
        print(p)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()

    if os.path.exists(os.path.join('ckpt', 'checkpoint')):
        model_file = tf.train.latest_checkpoint('ckpt')
        print("Restoring model weights from ", model_file)
        saver.restore(sess, model_file)
    start_time = time.time()

    train(model, sess, batch_num, args.epoch)
    test(model, sess, dataset_name, test_num, t)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.4f} seconds")


if __name__ == "__main__":
    main()
