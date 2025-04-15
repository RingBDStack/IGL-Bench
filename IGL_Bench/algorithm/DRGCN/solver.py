import time, random, heapq, os
import numpy as np
import scipy.sparse as sp
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, roc_auc_score, roc_curve, auc)
import tensorflow_probability as tfp
os.environ['CUDA_VISIBLE_DEVICES'] = ''

tfd = tfp.distributions

from IGL_Bench.algorithm.DRGCN.load_data import data_process
import IGL_Bench.algorithm.DRGCN.models.graph as mg
import IGL_Bench.algorithm.DRGCN.models.gmm as gmm
import IGL_Bench.algorithm.DRGCN.models.adversarialNets as ma
import IGL_Bench.algorithm.DRGCN.sparse as sparse
from IGL_Bench.algorithm.DRGCN.sparse import sparse_to_tuple
from sklearn.metrics import (accuracy_score, f1_score, roc_curve, auc,
                             balanced_accuracy_score)

tf.disable_v2_behavior()

tfd = tfp.distributions

def get_tf_device():
    try:
        if tf.test.is_gpu_available():
            return '/gpu:0'
    except:
        pass
    return '/cpu:0'

class DRGCN_node_solver:
    @staticmethod
    def pairwise_l2_norm2(x, y):
        with tf.name_scope('pairwise_l2_norm2'):
            sx = tf.shape(x)[0]
            sy = tf.shape(y)[0]
            xx = tf.tile(tf.expand_dims(x, -1), [1, 1, sy])
            yy = tf.tile(tf.expand_dims(y, -1), [1, 1, sx])
            yy = tf.transpose(yy, [2, 1, 0])
            return tf.sqrt(tf.reduce_sum(tf.square(xx - yy), axis=1))

    def __init__(self, cfg, pyg_dataset, device: str = 'cuda:0'):
        device = get_tf_device()
        self.cfg = cfg
        seed = 123
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)

        (x, _, adj_norm, label_vec,
         tr_idx, te_idx, va_idx,
         real_gan_nodes, _, adj_neighbor,
         all_neighbor_nodes) = data_process(pyg_dataset)

        if label_vec.ndim == 1:
            self.C = int(label_vec.max()) + 1
            label_mat = np.eye(self.C, dtype=np.float32)[label_vec]
        else:
            label_mat = label_vec.astype(np.float32)
            self.C = label_mat.shape[1]

        self.x = x.astype(np.float32)
        self.adj_norm = adj_norm.astype(np.float32)
        self.labels = label_mat
        self.train_idx, self.test_idx, self.val_idx = tr_idx, te_idx, va_idx
        self.real_gan_nodes = real_gan_nodes
        self.adj_neighbor = adj_neighbor.astype(np.int32)
        self.all_neighbor_nodes = np.array(all_neighbor_nodes, dtype=np.int32)

        self.N, self.F = self.x.shape
        self.batch_size = max(1, len(tr_idx) // 2)
        self.batch_size_dist = len(tr_idx)
        self.adj_neighbor_num = adj_neighbor.shape[1]

        self.adj_norm_tuple = sparse.sparse_to_tuple(sp.coo_matrix(self.adj_norm))
        self.x_tuple = sparse.sparse_to_tuple(sp.coo_matrix(self.x))

        with tf.device(device):
            self._build_graph()
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

    def _build_graph(self):
        cfg = self.cfg
        self.ph = {
            'adj_norm': tf.sparse_placeholder(tf.float32, name='adj_norm'),
            'x': tf.sparse_placeholder(tf.float32, name='features'),
            'labels': tf.placeholder(tf.float32, name='node_labels'),
            'mask': tf.placeholder(tf.int32, shape=(self.N,))
        }
        self.holders = {
            'dropout_prob': tf.placeholder_with_default(0., shape=()),
            'num_features_nonzero': tf.placeholder(tf.int32)
        }
        self.adj_neighbor_batch = tf.placeholder(tf.int32,
                                                 shape=(self.batch_size,
                                                        self.adj_neighbor_num))
        self.gan_idx = tf.placeholder(tf.int32, shape=(self.batch_size,))
        self.gan_y = tf.placeholder(tf.float32, shape=[None, self.C])
        self.gan_z = tf.placeholder(tf.float32, shape=[None, cfg.noise_dim])

        gcn_fc1 = mg.GraphConvLayer(input_dim=self.F, output_dim=20,
                                     name='nn_fc1', holders=self.holders,
                                     act=tf.nn.relu, dropout=True)(adj_norm=self.ph['adj_norm'],
                                                                   x=self.ph['x'],
                                                                   sparse=True)
        self.nn_dl = mg.GraphConvLayer(input_dim=20, output_dim=self.C,
                                       name='nn_dl', holders=self.holders,
                                       act=tf.nn.softmax, dropout=True)(adj_norm=self.ph['adj_norm'],
                                                                        x=gcn_fc1)

        labeled_samples = tf.gather(self.nn_dl, self.train_idx)
        unlabeled_samples = tf.gather(self.nn_dl, tf.range(self.N))
        self.gmm_labeled = gmm.gaussianMixtureModel(1, self.C).make_mixture_posterior(labeled_samples)
        self.gmm_unlabeled = gmm.gaussianMixtureModel(1, self.C).make_mixture_prior()

        gan_x = tf.gather(self.nn_dl, self.gan_idx)
        neighbor_x = tf.gather(self.nn_dl, self.all_neighbor_nodes)

        G = ma.Generator(x_dim=self.C, y_dim=self.C, z_dim=cfg.noise_dim,
                          h_dim=cfg.hidden_dim)
        self.G_sample = G(self.gan_z, self.gan_y)
        with tf.variable_scope('gan/discriminator', reuse=tf.AUTO_REUSE):
            D_real, D_logit_real = ma.Discriminator(x_dim=self.C, y_dim=self.C,
                                                    h_dim=cfg.hidden_dim)(gan_x, self.gan_y)
            D_fake, D_logit_fake = ma.Discriminator(x_dim=self.C, y_dim=self.C,
                                                    h_dim=cfg.hidden_dim)(self.G_sample, self.gan_y)

        def masked_softmax_xent(preds, labels, mask):
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
            mask = tf.cast(mask, tf.float32)
            mask /= tf.reduce_mean(mask)
            loss *= mask
            return tf.reduce_mean(loss)

        loss_gcn = masked_softmax_xent(self.nn_dl, self.ph['labels'], self.ph['mask'])
        kl_loss = tf.reduce_mean(tfd.kl_divergence(self.gmm_labeled, self.gmm_unlabeled))

        l2_dist = DRGCN_node_solver.pairwise_l2_norm2(self.G_sample, neighbor_x)
        mask_f = tf.cast(self.adj_neighbor_batch, tf.float32)
        mask_f /= tf.reduce_mean(mask_f)
        l2_loss = tf.reduce_mean(l2_dist * mask_f)

        D_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real,
                                                              labels=tf.ones_like(D_logit_real))
        D_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake,
                                                              labels=tf.zeros_like(D_logit_fake))
        loss_d = tf.reduce_mean(D_loss_real + D_loss_fake) + l2_loss
        loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake,
                                                                        labels=tf.ones_like(D_logit_fake))) + l2_loss

        lr = cfg.learning_rate
        self.gcn_train_op = tf.train.AdamOptimizer(lr).minimize(0.3 * loss_gcn + 0.7 * kl_loss)
        self.D_train_op = tf.train.AdamOptimizer(lr).minimize(loss_d)
        self.G_train_op = tf.train.AdamOptimizer(lr).minimize(loss_g)

        correct = tf.equal(tf.argmax(self.nn_dl, 1), tf.argmax(self.ph['labels'], 1))
        acc_mask = tf.cast(self.ph['mask'], tf.float32)
        acc_mask /= tf.reduce_mean(acc_mask)
        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32) * acc_mask)

        self.loss_gcn, self.kl_loss = loss_gcn, kl_loss
        self.loss_d, self.loss_g = loss_d, loss_g
        self.neighbor_x = neighbor_x

    def _mask_vec(self, idx):
        m = np.zeros(self.N, dtype=np.int32)
        m[idx] = 1
        return m

    def _base_feed(self, dropout):
        return {self.ph['adj_norm']: self.adj_norm_tuple,
                self.ph['x']: self.x_tuple,
                self.holders['dropout_prob']: dropout,
                self.holders['num_features_nonzero']: self.x_tuple[1].shape}

    def _sample_Z(self, m):
        return np.random.uniform(0., 1., size=[m, self.cfg.noise_dim]).astype(np.float32)

    def _gan_batch(self, epoch):
        total = len(self.real_gan_nodes)
        begin = (epoch * self.batch_size) % total
        batch_size = self.batch_size if total - begin >= self.batch_size else total - begin
        sel = [self.real_gan_nodes[(begin + i) % total] for i in range(batch_size)]
        idx = np.array([s[0] for s in sel], dtype=np.int32)
        lab = np.array([s[1] for s in sel], dtype=np.int32)
        neigh = self.adj_neighbor[begin:begin + batch_size]
        if batch_size < self.batch_size:
            pad_size = self.batch_size - batch_size
            idx = np.pad(idx, (0, pad_size), mode='wrap')
            lab = np.pad(lab, (0, pad_size), mode='wrap')
            neigh = np.pad(neigh, ((0, pad_size), (0, 0)), mode='wrap')
        return idx, lab, neigh

    def train(self, epochs: int = 500, log_every: int = 1, patience: int = 10):
        self.reset_parameters()
        t0 = time.time()
        best_val_acc = 0.0
        epochs_without_improvement = 0
        for ep in range(epochs):
            feed_gcn = self._base_feed(self.cfg.dropout_prob)
            feed_gcn.update({self.ph['labels']: self.labels,
                            self.ph['mask']: self._mask_vec(self.train_idx)})
            _, lg, kl, acc = self.sess.run([self.gcn_train_op, self.loss_gcn,
                                            self.kl_loss, self.accuracy], feed_dict=feed_gcn)
            z = self._sample_Z(self.batch_size)
            b_idx, b_lab, b_nei = self._gan_batch(ep) 
            y_fb = np.eye(self.C, dtype=np.float32)[b_lab]
            feed_gan = self._base_feed(0.)
            feed_gan.update({self.gan_z: z, self.gan_idx: b_idx, self.gan_y: y_fb,
                            self.adj_neighbor_batch: b_nei})
            _, ld = self.sess.run([self.D_train_op, self.loss_d], feed_dict=feed_gan)
            _, lg_ = self.sess.run([self.G_train_op, self.loss_g], feed_dict=feed_gan)
            if ep % log_every == 0:
                val_acc = self._eval_split(self.val_idx)[0]
                print(f"Ep {ep:03d} | gcn {lg:.4f} | kl {kl:.4f} | D {ld:.4f} | G {lg_:.4f} | "
                    f"val_acc {val_acc:.4f} | t {time.time()-t0:.1f}s")
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping triggered after {ep+1} epochs with no improvement.")
                    break

    def reset_parameters(self):
        if hasattr(self, 'sess'):
            self.sess.close()
        tf.reset_default_graph()
        with tf.device(get_tf_device()):
            self._build_graph()
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

    def _eval_split(self, idx: np.ndarray):
        mask = self._mask_vec(idx)
        feed = self._base_feed(0.)
        feed.update({self.ph['labels']: self.labels,
                     self.ph['mask']: mask})
        preds = self.sess.run(self.nn_dl, feed_dict=feed)
        true = self.labels[idx].argmax(1)
        pred_cls = preds[idx].argmax(1)
        acc = accuracy_score(true, pred_cls)
        bacc = balanced_accuracy_score(true, pred_cls)
        mf1 = f1_score(true, pred_cls, average='macro')
        fpr, tpr, _ = roc_curve(self.labels[idx].ravel(), preds[idx].ravel())
        roc_auc = auc(fpr, tpr)
        return acc, bacc, mf1, roc_auc

    def test(self):
        return self._eval_split(self.test_idx)
