'''
A logistic regression learning algorithm example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function
import sys
sys.path.append("/data")
from datetime import datetime
import tensorflow as tf
import numpy as np

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

"""inserted code"""
from scripts.utils.tf_utils import GradientSearcher

gradient_search = GradientSearcher()
"""inserted code"""

# Parameters
learning_rate = 0.01
training_epochs = 250
batch_size = 100
display_step = 1

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 784])  # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 10])  # 0-9 digits recognition => 10 classes

# Set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Construct model
pred = tf.nn.softmax(tf.matmul(x, W) + b)  # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))

"""insert code"""
obj_function = tf.reduce_min(tf.abs(pred))
obj_grads = tf.gradients(obj_function, x)[0]

# exp/expm1
# obj_function = -1* tf.reduce_max(target)

# div/log/reciprocal/rsqrt
# obj_function = tf.reduce_min(tf.abs(pred))

# log1p/sqrt
# obj_function = tf.reduce_min(target)
"""insert code"""

# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

"""insert code"""
# change part
batch_xs, batch_ys = mnist.train.next_batch(100)
max_val, min_val = np.max(batch_xs), np.min(batch_xs)
gradient_search.build(batch_size=batch_size, min_val=min_val, max_val=max_val)
"""insert code"""

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            # batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)

            """inserted code"""
            monitor_vars = {'loss': cost, 'obj_function': obj_function, 'obj_grad': obj_grads}
            feed_dict = {x: batch_xs, y: batch_ys}
            # if it's autoencoder
            # feed_dict is like feed_dict = {x: batch_xs}
            batch_xs, scores_rank = gradient_search.update_batch_data(session=sess, monitor_var=monitor_vars,
                                                                      feed_dict=feed_dict,
                                                                      iter_counter=epoch * total_batch + i,
                                                                      input_data=batch_xs)
            """inserted code"""

            _, c = sess.run([optimizer, cost], feed_dict=feed_dict)

            """inserted code"""
            new_batch_xs, new_batch_ys = mnist.train.next_batch(100)
            new_data_dict = {'x': new_batch_xs, 'y': new_batch_ys}
            old_data_dict = {'x': batch_xs, 'y': batch_ys}
            batch_xs, batch_ys = gradient_search.switch_new_data(new_data_dict=new_data_dict,
                                                                 old_data_dict=old_data_dict, scores_rank=scores_rank)
            # if it's autoencoder (unsupervised)
            # new_batch_xs = get_next_batch_x(batch_size)
            # new_data_dict = {'x': new_batch_xs}
            # old_data_dict = {'x': batch_xs}
            # batch_xs, _ = gradient_search.switch_new_data(new_data_dict=new_data_dict,
            #                                                      old_data_dict=old_data_dict, scores_rank=scores_rank)

            """inserted code"""

    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    # print("Not Found!", datetime.now() - s1)
    gradient_search.end_with_time("Not Found!")
