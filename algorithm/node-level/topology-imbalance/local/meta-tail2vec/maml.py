import numpy as np
import tensorflow as tf
import GPUtil


class MAML:
    def __init__(self, d, meta_lr=0.001, train_lr=0.01):
        self.d = d
        self.meta_lr = meta_lr
        self.train_lr = train_lr

        print('embedding shape:', self.d, 'meta-lr:', meta_lr, 'train-lr:', train_lr)

    def build(self, support_nb, support_xb, support_yb, query_nb, query_xb, query_yb, k, meta_batchsz, mode='train'):
        self.weights = self.conv_weights()
        training = True if mode is 'train' else False

        def meta_task(input):
            support_n, support_x, support_y, query_n, query_x, query_y = input
            query_preds, query_losses, query_nodes = [], [], []
            support_pred = self.forward(support_x, self.weights)
            if training:
                support_loss = tf.losses.mean_squared_error(support_y, support_pred)
            else:
                idx = tf.reshape(tf.where(tf.reshape(support_n[0], [-1]) > 0), [-1])
                support_loss = tf.losses.mean_squared_error(tf.gather(support_y, idx), tf.gather(support_pred, idx))

            grads = tf.gradients(support_loss, list(self.weights.values()))
            gvs = dict(zip(self.weights.keys(), grads))

            fast_weights = dict(
                zip(self.weights.keys(), [self.weights[key] - self.train_lr * gvs[key] for key in self.weights.keys()]))
            query_pred = self.forward(query_x, fast_weights)
            query_loss = tf.losses.mean_squared_error(query_y, query_pred)
            query_pred = tf.reshape(query_pred, [-1])
            query_n = tf.reshape(query_n, [-1])
            query_preds.append(query_pred)
            query_nodes.append(query_n)
            query_losses.append(query_loss)

            for _ in range(1, k):
                if training:
                    loss = tf.losses.mean_squared_error(support_y, self.forward(support_x, fast_weights))
                else:
                    idx = tf.reshape(tf.where(tf.reshape(support_n[0], [-1]) > 0), [-1])
                    loss = tf.losses.mean_squared_error(tf.gather(support_y, idx),
                                                        tf.gather(self.forward(support_x, fast_weights), idx))
                grads = tf.gradients(loss, list(fast_weights.values()))
                gvs = dict(zip(fast_weights.keys(), grads))
                fast_weights = dict(zip(fast_weights.keys(), [fast_weights[key] - self.train_lr * gvs[key]
                                                              for key in fast_weights.keys()]))
                query_pred = self.forward(query_x, fast_weights)
                query_loss = tf.losses.mean_squared_error(query_y, query_pred)
                query_pred = tf.reshape(query_pred, [-1])
                query_n = tf.reshape(query_n, [-1])
                query_preds.append(query_pred)
                query_nodes.append(query_n)
                query_losses.append(query_loss)

            result = [support_pred, support_loss, query_preds, query_losses, query_nodes]
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                print(f"total{gpu.memoryTotal} used memory {gpu.memoryUsed}MB free{gpu.memoryFree}")
            return result

        out_dtype = [tf.float32, tf.float32, [tf.float32] * k, [tf.float32] * k, [tf.float32] * k]
        result = tf.map_fn(meta_task, elems=(support_nb, support_xb, support_yb, query_nb, query_xb, query_yb),
                           dtype=out_dtype, name='map_fn')
        support_pred_tasks, support_loss_tasks, query_preds_tasks, query_losses_tasks, query_nodes = result

        if mode is 'train':
            self.support_loss = support_loss = tf.reduce_sum(support_loss_tasks) / meta_batchsz
            self.query_losses = query_losses = [tf.reduce_sum(query_losses_tasks[j]) / meta_batchsz
                                                for j in range(k)]

            optimizer = tf.train.AdamOptimizer(self.meta_lr, name='meta_optim')
            gvs = optimizer.compute_gradients(self.query_losses[-1])
            gvs = [(tf.clip_by_norm(grad, 10), var) for grad, var in gvs]
            self.meta_op = optimizer.apply_gradients(gvs)

        else:
            self.test_support_loss = support_loss = tf.reduce_sum(support_loss_tasks) / meta_batchsz
            self.test_query_losses = query_losses = [tf.reduce_sum(query_losses_tasks[j]) / meta_batchsz
                                                     for j in range(k)]
            self.test_query_preds = query_preds_tasks
            self.query_nodes = query_nodes

        tf.summary.scalar(mode + '：support loss', support_loss)
        for j in range(k):
            tf.summary.scalar(mode + '：query loss, step ' + str(j + 1), query_losses[j])

    def conv_weights(self):
        weights = {}
        fc_initializer = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope('MAML', reuse=tf.AUTO_REUSE):
            weights['w1'] = tf.get_variable('w1', [self.d, 1024], initializer=fc_initializer)
            weights['b1'] = tf.get_variable('b1', initializer=tf.zeros([1024]))
            weights['out_w'] = tf.get_variable('out_w', [1024, self.d], initializer=fc_initializer)
            weights['out_b'] = tf.get_variable('out_b', initializer=tf.zeros([self.d]))

        return weights

    def forward(self, x, weights):
        layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['w1']), weights['b1']))
        out_layer = tf.add(tf.matmul(layer_1, weights['out_w']), weights['out_b'])
        return out_layer
