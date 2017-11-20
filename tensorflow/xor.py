#!/usr/bin/env python

import numpy as np
import tensorflow as tf

# Weights for layer 1
w1 = tf.get_variable("W1", initializer=np.array([[1.0, 1.0],
                                                 [1.0, 1.0]]))

# Bias for layer 1
b1 = tf.get_variable("b1", initializer=np.array([0.0, -1.0]))

# Weights for layer 2
w2 = tf.get_variable("W2", initializer=np.array([[1.0], [-2.0]]))

# Bias for layer 2
b2 = tf.get_variable("b2", initializer=np.array([0.0]))

# Placeholder variable for inputs
x = tf.placeholder(tf.float64)

# Definition of hidden layer
hidden_layer = tf.nn.relu(b1 + tf.matmul(x, w1))

# Definition of output layer
y = b2 + tf.matmul(hidden_layer, w2)

# Run network on XOR inputs
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    x_in = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_out = sess.run(y, feed_dict={x:x_in})

# Print network response
print("{:8<} : {}".format("Input", "Output"))
for x_, y_ in zip(x_in, y_out):
    print("{:8<} : {}".format(x_, np.squeeze(y_)))
