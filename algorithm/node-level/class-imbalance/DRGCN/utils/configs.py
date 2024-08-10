# -*- coding: utf-8 -*-

import tensorflow as tf

# define the paths of the datasets
tf.flags.DEFINE_string("cora", "../dataset/cora", "")


tf.flags.DEFINE_integer("noise_dim", 20, "Dimensionality of generated noise in GAN")
tf.flags.DEFINE_integer("hidden_dim", 100, "Dimensionality of hidden networks in GAN")

# tf.flags.DEFINE_integer("hidden_size_gcn", 20, "Dimensionality of GCN hidden")

tf.flags.DEFINE_float('emb_dropout_prob', 0.2, 'Dropout probability of embedding layer')
tf.flags.DEFINE_float('dropout_prob', 0.3, 'Dropout probability of output layer')

tf.flags.DEFINE_float('learning_rate', 2e-3, 'Initial learning rate.')
tf.flags.DEFINE_float('learning_rate_gan', 5e-4, 'Initial learning rate.')
tf.flags.DEFINE_float('weight_decay', 3e-2, 'Weight for L2 loss on embedding matri')
tf.flags.DEFINE_float('early_stopping', 10, 'allow elary stop')

FILES = tf.flags.FLAGS