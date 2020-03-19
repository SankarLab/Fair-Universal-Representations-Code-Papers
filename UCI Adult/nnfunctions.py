import tensorflow as tf
import numpy as np
def leakyRelu(inputs, rate=0.01): # I have also changed the rate here from 0.1 to 0.01.
    return tf.maximum(inputs, tf.minimum(rate*inputs, 0))

def fc_bn_leakyRelu(inputs, numOutput, alpha=0.01, keep_prob = 1.0):   # I am changing this alpha to 0.01, you might also try to change the drop out thing!
    fc=tf.contrib.layers.fully_connected(inputs, numOutput,
                                         weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                         weights_regularizer=tf.contrib.layers.l2_regularizer(0.1),
                                         activation_fn=tf.identity, scope=None )
    fc=leakyRelu(fc, rate=alpha)
    fc_dropout = tf.nn.dropout(fc, keep_prob=keep_prob)
    fc=tf.contrib.layers.batch_norm(fc_dropout)

    return fc


def weight_variable(shape):
    initial=tf.truncated_normal(shape, stddev=0.02)
    return tf.Variable(initial, dtype=tf.float32)

def bias_variable(shape):
    initial=tf.constant(0.1, shape=shape)
    return tf.Variable(initial, dtype=tf.float32)

#def noise_projection(input, weights):


def conv2d (inputs, weights):
    return tf.nn.conv2d(inputs, weights, strides = [1,1,1,1], padding='SAME')

def deconv2d (inputs, weights):
    return tf.nn.conv2d_transpose(inputs, weights, strides=[1,1,1,1], padding='SAME')

def maxpool2d (inputs):
    return tf.nn.max_pool(inputs, ksize = [1,2,2,1], strides = [1,2,2,1], padding='SAME')
