import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import logging
import os
import sys
sys.path.append("/data")
from datetime import datetime


def get_logger(filepath, level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    # create a file handler
    handler = logging.FileHandler(filepath)
    handler.setLevel(logging.INFO)

    # create a logging format
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)
    return logger


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


def random_data(row, column):
    return np.random.uniform(-1., 1., size=[row, column])


def weight_variable(shape, stddev=0.1):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)


def bias_variable(shape, bais=0.1):
    initial = tf.constant(bais, shape=shape)
    return tf.Variable(initial)


# 鉴别网络weights
d_w1 = weight_variable([784, 128])
d_b1 = bias_variable([128])

d_w2 = weight_variable([128, 1])
d_b2 = bias_variable([1])

param_d = [d_w1, d_w2, d_b1, d_b2]

# 生成网络weights
g_w1 = weight_variable([100, 128])
g_b1 = bias_variable([128])

g_w2 = weight_variable([128, 784])
g_b2 = bias_variable([784])

param_g = [g_w1, g_w2, g_b1, g_b2]


# 鉴别网络
def d_network(x):
    d1 = tf.nn.relu(tf.matmul(x, d_w1) + d_b1)
    d_out = tf.matmul(d1, d_w2) + d_b2
    return tf.nn.sigmoid(d_out)


# 生成网络
def g_network(x):
    g1 = tf.nn.relu(tf.matmul(x, g_w1) + g_b1)
    g_out = tf.matmul(g1, g_w2) + g_b2
    return tf.nn.sigmoid(g_out)


x = tf.placeholder(tf.float32, shape=[None, 784])
z = tf.placeholder(tf.float32, shape=[None, 100])

d_out_real = d_network(x)

g_out = g_network(z)
d_out_fake = d_network(g_out)

#MUTATION#
d_loss = -tf.reduce_mean(tf.log(d_out_real) + tf.log(1. - d_out_fake + 1e-8))
g_loss = -tf.reduce_mean(tf.log(d_out_fake + 1e-8))

d_optimizer = tf.train.AdamOptimizer().minimize(d_loss, var_list=param_d)
g_optimizer = tf.train.AdamOptimizer().minimize(g_loss, var_list=param_g)

batch_size = 256
max_step = 1000000
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

"""insert code"""
from scripts.utils.tf_utils import GradientSearcher
gradient_search = GradientSearcher(name="gan_practice_grist")
obj_function = tf.reduce_min(tf.abs(d_out_real))
obj_grads = tf.gradients(obj_function, x)[0]
batch_real, _y = mnist.train.next_batch(batch_size)
max_val, min_val = np.max(batch_real), np.min(batch_real)
gradient_search.build(batch_size=batch_size,min_val=min_val,max_val=max_val)
"""insert code"""

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("training")
    i = 0
    step = 0
    while True:
        """inserted code"""
        monitor_vars = {'loss': d_loss, 'obj_function': obj_function, 'obj_grad': obj_grads}
        feed_dict = {x: batch_real, z: random_data(batch_size, 100)}
        batch_real, scores_rank = gradient_search.update_batch_data(session=sess, monitor_var=monitor_vars,
                                                                  feed_dict=feed_dict,input_data=batch_real)
        """inserted code"""

        _, d_loss_train = sess.run([d_optimizer, d_loss], feed_dict=feed_dict)

        """inserted code"""
        new_batch_xs, new_batch_ys = mnist.train.next_batch(batch_size)
        new_data_dict = {'x': new_batch_xs}
        old_data_dict = {'x': batch_real}
        batch_real, _ = gradient_search.switch_new_data(new_data_dict=new_data_dict, old_data_dict=old_data_dict,
                                                             scores_rank=scores_rank)

        """inserted code"""

        _, g_loss_train = sess.run([g_optimizer, g_loss], feed_dict={z: random_data(batch_size, 100)})

        if step % 1000 == 0:
            samples = sess.run(g_out, feed_dict={z: random_data(16, 100)})
            print("step %s: d_loss is %s, gan_loss is %s" % (step, d_loss_train, g_loss_train))
            print("step %s: d_loss is %s, g_loss is %s" % (step, d_loss_train, g_loss_train))
        step += 1
        gradient_search.check_time()