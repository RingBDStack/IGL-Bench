import time
import tensorflow as tf
import numpy as np
import sys 
sys.path.append('..')
import models.graph as mg
import models.gmm as g
import models.adversarialNets as ma
import scipy.sparse
from utils import data_process, sparse
from utils import configs, metrics
import tensorflow_probability as tfp

tfd = tfp.distributions

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

train_ratio = 0.03
lrate_gcn = configs.FILES.learning_rate
dataset = configs.FILES.cora
print("train_ratio:",train_ratio)

x, _, adj_norm, labels, train_indexes, test_indexes, validation_indexes, real_gan_nodes, _, adj_neighbor, all_neighbor_nodes = data_process.load_data_boundary(dataset, str(train_ratio), 
                                                                                x_flag='feature')
batch_size = int(len(train_indexes) / 2)
batch_size_dist = len(train_indexes)
node_num = adj_norm.shape[0]
label_num = labels.shape[1]
adj_neighbor_num = adj_neighbor.shape[1]


adj_norm_tuple = sparse.sparse_to_tuple(scipy.sparse.coo_matrix(adj_norm))
feat_x_nn_tuple = sparse.sparse_to_tuple(scipy.sparse.coo_matrix(x))

# node-node network train and validate masks
nn_train_mask = np.zeros([node_num,])
nn_validation = np.zeros([node_num,])
nn_test_mask = np.zeros([node_num,])

for i in train_indexes:
    nn_train_mask[i] = 1
    
for i in validation_indexes:
    nn_validation[i] = 1
    
for i in test_indexes:
    nn_test_mask[i] = 1

# labeled nodes for generating corresponding fake nodes 
real_node_idx = []
real_node_lab = []
for chunk in real_gan_nodes:
    real_node_idx.append(int(chunk[0]))
    real_node_lab.append(int(chunk[1]))
y_fb = np.eye(label_num)[real_node_lab]
print('real_node_idx:{}', real_node_idx)
print('real_node_lab:{}', real_node_lab)
    
# TensorFlow placeholder for GCN
ph = {
      'adj_norm': tf.sparse_placeholder(tf.float32, name="adj_norm"),
      'x': tf.sparse_placeholder(tf.float32, name="features"),
      'labels': tf.placeholder(tf.float32, name="node_labels"),
      'mask': tf.placeholder(tf.int32, shape=(node_num,))
      }

placeholders = {
                'dropout_prob': tf.placeholder_with_default(0., shape=()),
                'num_features_nonzero': tf.placeholder(tf.int32)
                }
# TensorFlow placeholder for GAN
# gan_x = tf.placeholder(tf.float32, shape=[None, label_num])
adj_neighbor_batch = tf.placeholder(tf.int32, shape=(batch_size, adj_neighbor_num))
real_neighbor_x = tf.placeholder(tf.float32, shape=[None, label_num])
real_sample_x = tf.placeholder(tf.float32, shape=[None, label_num])
gan_idx = tf.placeholder(tf.int32, shape=(batch_size,))
gan_y = tf.placeholder(tf.float32, shape=[None, label_num])
gan_z = tf.placeholder(tf.float32, shape=[None, configs.FILES.noise_dim])

# placeholder for the prior gaussian mixture model
loc_labeled = tf.placeholder(tf.float32, shape=[label_num, label_num])
scale_diag_labeled = tf.placeholder(tf.float32, shape=[label_num, label_num])
mix_dist = tf.placeholder(tf.float32, shape=(label_num,))
unlabeled_sample_batch = tf.placeholder(tf.float32, shape=[None, label_num])
unlabeled_sample_idx = tf.placeholder(tf.int32, shape=(batch_size_dist,))

# define the GCN representation learning model (2-layer)
t_model = mg.GraphConvLayer(input_dim=x.shape[-1],
                           output_dim=10,
                           name='nn_fc1',
                           holders=placeholders,
                           act=tf.nn.relu,
                           dropout=True)

nn_fc1 = t_model(adj_norm=ph['adj_norm'],
                           x=ph['x'], sparse=True)
                            
nn_dl = mg.GraphConvLayer(input_dim=10,
                           output_dim=label_num,
                           name='nn_dl',
                           holders=placeholders,
                           act=tf.nn.softmax,
                           dropout=True)(adj_norm=ph['adj_norm'],
                                           x=nn_fc1)

# samples for training the Gaussian mixture models
labeled_samples = tf.gather(nn_dl, train_indexes)
unlabeled_samples = tf.gather(nn_dl, unlabeled_sample_idx)
 
gmm_labeled = g.gaussianMixtureModel(1, label_num).make_mixture_posterior(labeled_samples)
#  
# gmm_labeled = g.gaussianMixtureModel(1, label_num).make_mixture_prior(loc=loc_labeled,
#                                                                               raw_scale_diag=scale_diag_labeled,
#                                                                               mixture_logits=mix_dist)
gmm_unlabeled = g.gaussianMixtureModel(1, label_num).make_mixture_prior()

# from sklearn.mixture import GaussianMixture as GMM
# tf.constant(labeled_samples)
                           
# define the GAN pseudo samples generation model
gan_x = tf.gather(nn_dl, gan_idx)
neighbor_x = tf.gather(nn_dl, all_neighbor_nodes)

g_model = ma.Generator(x_dim=label_num,
                        y_dim=label_num,
                        z_dim=configs.FILES.noise_dim,
                        h_dim=configs.FILES.hidden_dim)
G_sample = g_model(gan_z, gan_y)
                        
D_real, D_logit_real = ma.Discriminator(x_dim=label_num,
                                        y_dim=label_num,
                                        h_dim=configs.FILES.hidden_dim)(gan_x, gan_y)
D_fake, D_logit_fake = ma.Discriminator(x_dim=label_num,
                                        y_dim=label_num,
                                        h_dim=configs.FILES.hidden_dim)(G_sample, gan_y)


def pairwise_l2_norm2(x, y):
    with tf.op_scope([x, y],'pairwise_l2_norm2'):
        size_x = tf.shape(x)[0]
        size_y = tf.shape(y)[0]
        xx = tf.expand_dims(x, -1)
        xx = tf.tile(xx, tf.stack([1, 1, size_y]))

        yy = tf.expand_dims(y, -1)
        yy = tf.tile(yy, tf.stack([1, 1, size_x]))
        yy = tf.transpose(yy, perm=[2, 1, 0])

        diff = tf.subtract(xx, yy)
        square_diff = tf.square(diff)

        square_dist = tf.reduce_sum(square_diff, 1)
        sqrt_dist = tf.sqrt(square_dist)

        return sqrt_dist

def kullback_dist(x, y):
    mul = tf.matmul(x, tf.transpose(y))
    probs = 1 / (1 + tf.exp(-mul))
    return probs

# calculate the combined loss for GCN
def masked_sigmoid_softmax_cross_entropy(preds, labels, mask):
    # loss for the GCN
    loss_gcn = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss_gcn *= mask
    for var in t_model.var.values():
        loss_gcn += configs.FILES.weight_decay * tf.nn.l2_loss(var)
        
    return tf.reduce_mean(loss_gcn)

# calculate the combined loss for GAN 
def masked_sigmoid_cross_entropy(D_real_preds, D_fake_preds,fake_sample_x, 
                                    real_labeled_x, adj_neighbor_x, regularized_mask):
    
    l2_distance = pairwise_l2_norm2(fake_sample_x, adj_neighbor_x)
    regularized_mask = tf.cast(regularized_mask, dtype=tf.float32) 
    regularized_mask /= tf.reduce_mean(regularized_mask)
    l2_loss = l2_distance * regularized_mask

    # kullback divergence to measure the empirical (from gcn learning) and generated (from generator) distributions
#     empirical_dist = kullback_dist(real_labeled_x, adj_neighbor_x)
#     generated_dist = kullback_dist(fake_sample_x, adj_neighbor_x)
#     kb_divergence = empirical_dist * tf.log(empirical_dist / generated_dist)
#     regularized_mask = tf.cast(regularized_mask, dtype=tf.float32) 
#     regularized_mask /= tf.reduce_mean(regularized_mask)
#     l2_loss = kb_divergence * regularized_mask
    
    # loss for discriminator
    D_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_preds, labels=tf.ones_like(D_real_preds))
    D_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_preds, labels=tf.zeros_like(D_fake_preds))
    loss_d = D_loss_real + D_loss_fake + l2_loss
#     for var in t_model.var.values():
#         loss_d += configs.FILES.weight_decay * tf.nn.l2_loss(var)

    # loss for generator
    loss_g = tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_preds, labels=tf.ones_like(D_fake_preds))
    loss_g = loss_g + l2_loss
#     for var in g_model.var.values():
#         loss_g += configs.FILES.weight_decay * tf.nn.l2_loss(var)
     
    return tf.reduce_mean(loss_d), tf.reduce_mean(loss_g)

def l2_norm_regularization(fake_sample_x, real_sample_x, adj_neighbor_x, 
                                regularized_mask):
    
    # loss for regularized generator
    l2_distance = pairwise_l2_norm2(fake_sample_x, adj_neighbor_x)
    regularized_mask = tf.cast(regularized_mask, dtype=tf.float32) 
    regularized_mask /= tf.reduce_mean(regularized_mask)
    l2_loss = l2_distance * regularized_mask
    
    return tf.reduce_mean(l2_loss)

def distribution_regularization(posterior_samples):
     
#     kl_loss = gmm_unlabeled.log_prob(posterior_samples) - gmm_labeled.log_prob(posterior_samples)
    kl_loss = tfd.kl_divergence(gmm_labeled, gmm_unlabeled)
    return tf.reduce_mean(kl_loss)
    
# calculate the classification accuracy for GCN
def masked_accuracy(preds, labels, mask):
    
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    
    return tf.reduce_mean(accuracy_all), correct_prediction

# calculate the classification accuracy per classes   
def precision_per_class(preds, labels, mask):
    import heapq
    mask = mask.astype(int)
    labels = labels.astype(int)
    val_indexes = np.where(mask==1)[0]
    pred_true_labels = {}
    for i in val_indexes:
        pred_probs_i = preds[i]
        true_raw_i = labels[i]
        
        pred_label_i = heapq.nlargest(np.sum(true_raw_i),range(len(pred_probs_i)), 
                                      pred_probs_i.take)
        true_label_i = np.where(true_raw_i==1)[0]
        pred_true_labels[i] = (pred_label_i, true_label_i)
    accuracy_per_classes = metrics.evaluate(pred_true_labels)
    
    from sklearn.metrics import roc_curve, auc
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    test_y = labels[val_indexes]
    test_pred = preds[val_indexes]
    fpr["micro"], tpr["micro"], _ = roc_curve(test_y.ravel(), test_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
#     print('micro_auc=',roc_auc["micro"])
    
    return accuracy_per_classes, roc_auc["micro"]

# define the collective optimization process for GCN and GAN
with tf.name_scope('optimizer'):
    
    loss_gcn = masked_sigmoid_softmax_cross_entropy(preds=nn_dl, 
                                            labels=ph['labels'], 
                                            mask=ph['mask'])
    
    loss_d, loss_g = masked_sigmoid_cross_entropy(D_real_preds=D_logit_real, D_fake_preds=D_logit_fake,
                                                  fake_sample_x=G_sample,
                                                  real_labeled_x=real_sample_x,
                                                  adj_neighbor_x=real_neighbor_x,
                                                  regularized_mask=adj_neighbor_batch)


    # KL-divergence regularization
    kl_loss = distribution_regularization(unlabeled_samples)
#     l2_loss = l2_norm_regularization(D_real_preds=D_logit_real,
#                                             D_fake_preds=D_logit_fake,
#                                             fake_sample_x=reg_sample_x,
#                                             real_neighbor_x=neighbor_x,
#                                             regularized_mask=adj_neighbor_batch)    
    # gcn optimizer
    gcn_opt = tf.train.AdamOptimizer(learning_rate=lrate_gcn).minimize(0.3*loss_gcn + 0.7*kl_loss)
    
    
    # gcn gan combined optimizer
#     gcn_gan_opt = tf.train.AdamOptimizer(learning_rate=lrate_gcn).minimize(gcn_gan_loss)
    # gan optimizer
#     reg_solver = tf.train.AdamOptimizer(learning_rate=lrate_gcn).minimize(l2_loss)
    D_solver = tf.train.AdamOptimizer(learning_rate=lrate_gcn).minimize(loss_d)
    G_solver = tf.train.AdamOptimizer(learning_rate=lrate_gcn).minimize(loss_g)

    # gcn classification accuracy
    accuracy, correct_prediction = masked_accuracy(preds=nn_dl, 
                               labels=ph['labels'], mask=ph['mask'])

def sample_Z(m, n):
    temp = np.random.uniform(0., 1., size=[m, n])
#     temp_norm = temp / np.sum(temp, axis=1, keepdims=True)
    return temp

# define the feed_dict for gcn
feed_dict_gcn = {ph['adj_norm']: adj_norm_tuple,
                      ph['x']: feat_x_nn_tuple,
                      placeholders['dropout_prob']: configs.FILES.dropout_prob,
                      placeholders['num_features_nonzero']: feat_x_nn_tuple[1].shape,
                      ph['labels']: labels.toarray(),
                      ph['mask']: nn_train_mask
#                       gan_y: y_fb,
#                       gan_idx: real_node_idx
                      }

feed_dict_gan_d = {ph['adj_norm']: adj_norm_tuple,
                       ph['x']: feat_x_nn_tuple,
                       placeholders['dropout_prob']: 0.,
                       placeholders['num_features_nonzero']: feat_x_nn_tuple[1].shape,
                       gan_y: y_fb,
                       gan_idx: real_node_idx
                      }
# feed_dict_gan_g_reg = {ph['adj_norm']: adj_norm_tuple,
#                        ph['x']: feat_x_nn_tuple,
#                        placeholders['dropout_prob']: 0.,
#                        placeholders['num_features_nonzero']: feat_x_nn_tuple[1].shape
#                       }
feed_dict_gan_g = {gan_y: y_fb,
                   ph['adj_norm']: adj_norm_tuple,
                       ph['x']: feat_x_nn_tuple,
                       placeholders['dropout_prob']: 0.,
                       placeholders['num_features_nonzero']: feat_x_nn_tuple[1].shape,
                       gan_y: y_fb,
                       gan_idx: real_node_idx}

feed_test_dict_gcn = {ph['adj_norm']: adj_norm_tuple,
                    ph['x']: feat_x_nn_tuple,
                    ph['labels']: labels.toarray(),
                    ph['mask']: nn_test_mask,
                    placeholders['dropout_prob']: 0.,
                    placeholders['num_features_nonzero']: feat_x_nn_tuple[1].shape,
                    gan_idx: real_node_idx
                    }


# generate mini-batch for gan
def get_batch(pos, node_idx_set, node_lab_set, adj_neighbor_set):
    batch_idx = []
    batch_lab = []
    batch_neighbor = []
    total_num = len(node_idx_set)
    begin = pos * batch_size
    end = pos * batch_size + batch_size
    for i in range(begin, end):
        idx = i % total_num
        batch_idx.append(node_idx_set[idx])
        batch_lab.append(node_lab_set[idx])
        batch_neighbor.append(adj_neighbor_set[idx])
    return batch_idx, batch_lab, np.array(batch_neighbor)

def get_batch_samples(pos, bsize, test_idx):
    batch_idx = []
    begin = pos * bsize
    end = pos * bsize + bsize
    total_num = len(test_idx)
    for i in range(begin, end):
        idx = i % total_num
        batch_idx.append(idx)
        
    return batch_idx
    

def GMM_prior(num_components, data):
    
    from sklearn import mixture
    gmm = mixture.GaussianMixture(n_components=num_components, max_iter=500,
                                  covariance_type='diag')
    gmm.fit(data)
    g_loc = gmm.covariances_
    g_scale_dig = np.sqrt(gmm.covariances_)
    g_mixture_logits = gmm.weights_
    
    return g_loc, g_scale_dig, g_mixture_logits
    
    

sess = tf.Session()
sess.run(tf.global_variables_initializer())
epochs = 400
iters = 2
save_every = 1    
t = time.time()


# forwardly derive the node representations from gcn
for epoch in range(epochs):
    
    
    _, train_loss, kl_loss_v, train_acc, train_nn_dl, labeled_samples_run, neighbor_x_val = sess.run((gcn_opt, loss_gcn, kl_loss, accuracy, nn_dl, labeled_samples, neighbor_x), feed_dict=feed_dict_gcn)
    
#   if epoch % 20 == 0:
#     batch_sample_idx = get_batch_samples(epoch, batch_size_dist, test_indexes)
#     g_loc, g_scale_dig, g_mixture_logits = GMM_prior(label_num, labeled_samples_run)
#     feed_dict_gcn.update(({loc_labeled: g_loc, scale_diag_labeled: g_scale_dig, mix_dist:g_mixture_logits, unlabeled_sample_idx:batch_sample_idx}))
#     _, kl_loss_v = sess.run((kl_opt, kl_loss), feed_dict=feed_dict_gcn)
    Z_sample = sample_Z(batch_size, configs.FILES.noise_dim)
    feed_dict_gan_g.update({gan_z: Z_sample})
    feed_dict_gan_d.update({gan_z: Z_sample})
    
    batch_idx, batch_lab, batch_neighbor = get_batch(epoch, real_node_idx, real_node_lab, adj_neighbor)
    y_fb = np.eye(label_num)[batch_lab]
    
#     neighbor_x_val = sess.run(neighbor_x, feed_dict=feed_dict_gan_g_reg)
    gan_x_val = sess.run(tf.gather(train_nn_dl, batch_idx))
    
#         feed_dict_gan_d.update(({gan_idx: batch_idx, gan_y: y_fb}))
    feed_dict_gan_d.update(({gan_idx: batch_idx, gan_y: y_fb, adj_neighbor_batch: batch_neighbor, real_neighbor_x: neighbor_x_val, real_sample_x:gan_x_val}))
    feed_dict_gan_g.update(({gan_idx: batch_idx, gan_y: y_fb, adj_neighbor_batch: batch_neighbor, real_neighbor_x: neighbor_x_val, real_sample_x:gan_x_val}))
    feed_test_dict_gcn.update(({gan_idx: batch_idx}))
    
    _, D_loss_curr = sess.run([D_solver, loss_d], feed_dict=feed_dict_gan_d)
    _, G_loss_curr = sess.run([G_solver, loss_g], feed_dict=feed_dict_gan_g)
#       
#         _, reg_loss_curr = sess.run([reg_solver, l2_loss], feed_dict=feed_dict_gan_g_reg)
      
#     # generated embeddings of fake nodes
    if epoch % save_every == 0:
        test_acc, test_nn_dl, test_gan_x = sess.run((accuracy, nn_dl, gan_x), feed_dict=feed_test_dict_gcn)
        train_accuracy_per_classes,_ = precision_per_class(train_nn_dl, labels.toarray(), nn_train_mask)
        test_accuracy_per_classes, auc = precision_per_class(test_nn_dl, labels.toarray(), nn_test_mask)
        
        samples = sess.run(G_sample, feed_dict={gan_z: Z_sample, gan_y:y_fb})
        
        print("Epoch:", '%04d' % (epoch + 1),
              "train_loss=", "{:.5f}".format(train_loss),
              "train_loss_d=", "{:.5f}".format(D_loss_curr),
              "train_loss_g=", "{:.5f}".format(G_loss_curr),
              "train_acc=", "{:.5f}".format(train_acc),
              "test_acc=", "{:.5f}".format(test_acc),
              "test_auc=", "{:.5f}".format(auc),
              "time=", "{:.5f}".format(time.time() - t))
        print("train_accuracy_per_classes:{}".format(train_accuracy_per_classes))
        print("test_accuracy_per_classes:{}".format(test_accuracy_per_classes))
