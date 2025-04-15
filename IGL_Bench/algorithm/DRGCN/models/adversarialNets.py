import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_v2_behavior()


def glorot(shape, name):
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    initializer = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.get_variable(name, initializer=initializer)

def zeros(shape, name):
    initializer = tf.zeros(shape, dtype=tf.float32)
    return tf.get_variable(name, initializer=initializer)


class Generator:
    def __init__(self, x_dim, y_dim, z_dim, h_dim):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self._build_model()

    def _build_model(self):
        with tf.variable_scope("gan/generator"):
            self.G_W1 = glorot([self.z_dim + self.y_dim, self.h_dim], name='G_W1')
            self.G_b1 = zeros([self.h_dim], name='G_b1')
            self.G_W2 = glorot([self.h_dim, self.x_dim], name='G_W2')
            self.G_b2 = zeros([self.x_dim], name='G_b2')

    def call(self, z, y):
        inputs = tf.concat([z, y], axis=1)
        h1 = tf.nn.relu(tf.matmul(inputs, self.G_W1) + self.G_b1)
        log_prob = tf.matmul(h1, self.G_W2) + self.G_b2
        prob = tf.nn.softmax(tf.nn.tanh(log_prob))
        return prob

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)


class Discriminator:
    def __init__(self, x_dim, y_dim, h_dim):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.h_dim = h_dim
        self._build_model()

    def _build_model(self):
        with tf.variable_scope("gan/discriminator"):
            self.D_W1 = glorot([self.x_dim + self.y_dim, self.h_dim], name='D_W1')
            self.D_b1 = zeros([self.h_dim], name='D_b1')
            self.D_W2 = glorot([self.h_dim, 1], name='D_W2')
            self.D_b2 = zeros([1], name='D_b2')

    def call(self, x, y):
        inputs = tf.concat([x, y], axis=1)
        h1 = tf.nn.relu(tf.matmul(inputs, self.D_W1) + self.D_b1)
        logit = tf.matmul(h1, self.D_W2) + self.D_b2
        prob = tf.nn.sigmoid(logit)
        return prob, logit

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)