# Initialization of weights options -

# Initialize all weights as zeros - No randomness
# Random distribution near zero - Activation function distortion

# Xavier Initialization - Optimal
# Draw weights from a distribution with zero mean and a specific variance

# Learning Rate - Defines step size during Gradient Descent

# Batch Size - Allow the use of Stochastic Gradient Descent -
# (Smaller - Less representative of data, Larger - Longer training time)

# Second-Order Behavior of Gradient Descent -
# Errors start off large, so larger steps at the beginning and smaller when moving closer to the minimum
# AdaGrad, RMSProp, ADAM (Best Optimizer, Automated Steps)

# Vanishing Gradients - Layers towards input will be affected less by error calculations -
# Initialization and Normalization help these issues

# Protection against Overfitting -
# L1/L2 Regularization (Penalty for larger weights)
# Dropout (Remove random neurons during training)
# Data Expansion (Artificially expand data by adding noise, tilt, white noise)

# MNIST Dataset
# Image flattening removes the 2D info

# Softmax Regression Approach
# Returns a list of values between 0 and 1 that add up to 1

# Softmax Activation Function

import tensorflow as tf
import logging
logging.getLogger().setLevel(logging.INFO)
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Placeholders

x = tf.placeholder(tf.float32, shape=[None, 784])

# Variables

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Create Graph Operations

y = tf.matmul(x, W) + b

# Loss function

y_true = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y))

# Optimizer

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train = optimizer.minimize(cross_entropy)

# Create Session

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(1000):
        batch_x, batch_y = mnist.train.next_batch(100)
        sess.run(train, feed_dict={x: batch_x, y_true: batch_y})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))

    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print(sess.run(acc, feed_dict={
          x: mnist.test.images, y_true: mnist.test.labels}))
