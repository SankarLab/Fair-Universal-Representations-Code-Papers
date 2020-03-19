import tensorflow as tf
import numpy as np

def leakyRelu(inputs, rate=0.1):
    return tf.maximum(inputs, tf.minimum(rate*inputs, 0))
# if input>0, return the input; if input <0, return rate*input

def fc_bn_leakyRelu(inputs, numOutput, alpha=0.1):
    fc=tf.contrib.layers.fully_connected(inputs, numOutput,
                                         weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                         weights_regularizer=tf.contrib.layers.l2_regularizer(0.1),
                                         activation_fn=tf.identity, scope=None )
    fc=leakyRelu(fc, rate=alpha)
    fc=tf.contrib.layers.batch_norm(fc) # normalized the output

    return fc
# numOutput: the dimension of outputs

def weight_variable(shape):
    initial=tf.truncated_normal(shape, stddev=0.02)
    return tf.Variable(initial, dtype=tf.float32)

def bias_variable(shape):
    initial=tf.constant(0.1, shape=shape)
    return tf.Variable(initial, dtype=tf.float32)

#def noise_projection(input, weights):

def conv2d (inputs, weights):
    return tf.nn.conv2d(inputs, weights, strides = [1,1,1,1], padding='SAME') # strides: sample-dimension, row-dimension, column-dimension (these two are spatial dimensions), how many layers the filter applied to (RGB, 3; Gret: 1)

def deconv2d (inputs, weights):
    return tf.nn.conv2d_transpose(inputs, weights, strides=[1,1,1,1], padding='SAME')

def maxpool2d (inputs):
    return tf.nn.max_pool(inputs, ksize = [1,2,2,1], strides = [1,2,2,1], padding='SAME')
