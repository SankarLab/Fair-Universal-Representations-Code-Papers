# CNN based privatizer

import tensorflow as tf
import scipy.io
import numpy as np
import time
import math
from nnfunctions import *
import matplotlib.pyplot as plt

mat = scipy.io.loadmat('genki.mat')
# dimension of each data sample
Dim = np.asscalar(mat['D'])
# K1: dimension of class of facial expression, K2 dimension of class of gender
K1 = np.asscalar(mat['K1'])
K2 = np.asscalar(mat['K2'])

Ntest = np.asscalar(mat['Ntest'])
Ntrain = np.asscalar(mat['Ntrain'])

Xtrain = mat['Xtrain']
Xtest = mat['Xtest']

# label of expression 1: smile, 0: not smile
Y1train = mat['y1_train'] - 1
Y1test = mat['y1_test'] - 1

# label of gender 1: female, 0: male
Y2train = mat['y2_train'] - 1
Y2test = mat['y2_test'] - 1

Ndata = Ntrain + Ntest
Xdata = np.hstack((Xtrain, Xtest))
Y1data = np.hstack((Y1train, Y1test))
Y2data = np.hstack((Y2train, Y2test))

Xdata = np.transpose(Xdata)
Y1data = np.transpose(Y1data)
Y2data = np.transpose(Y2data)

Xtrain = np.transpose(Xtrain)
Y1train = np.transpose(Y1train)
Y2train = np.transpose(Y2train)

Xtest = np.transpose(Xtest)
Y1test = np.transpose(Y1test)
Y2test = np.transpose(Y2test)

print(np.shape(Xdata))
print(np.shape(Y1data))
print(np.shape(Y2data))

mbsize = 200
num_iter = 20000
learning_ratep = 0.0002
learning_ratea = 0.0002
eps = 0
noise_seed_dim = 100  # 2
d =1

X = tf.placeholder(tf.float32, shape=[None, Dim])
Y = tf.placeholder(tf.float32, shape=[None, K1])
Y_onehot = tf.placeholder(tf.float32, shape=[None, K1])
Z = tf.placeholder(tf.float32, shape=[None, noise_seed_dim], name='Z')
noise_seed = tf.placeholder(tf.float32, shape=[None, noise_seed_dim])
keep_prob = tf.placeholder(tf.float32)
penalty_rate = tf.placeholder(tf.float32)



a_loss = []
p_loss = []
p_dist = []

##This function shuffles the indexes of samples and returns batch_size number of samples.
def nextbatch(batchsize, X, Y1, Y2, datalen):
    idx = np.arange(0, datalen)
    np.random.shuffle(idx)
    idx = idx[:batchsize]
    X_shuffled = [X[i, :] for i in idx]
    Y1_shuffled = [Y1[i, :] for i in idx]
    Y2_shuffled = [Y2[i, :] for i in idx]
    return np.asarray(X_shuffled), np.asarray(Y1_shuffled), np.asarray(Y2_shuffled)


##CNN based adversary network: this function contains implementation of CNN based adversary network which classifies the sensitive data.
def adversarycnnbn(inputs, reuse=False):
    with tf.variable_scope("adversary", reuse=reuse):
        x_image = tf.reshape(inputs, shape=[-1, 16, 16, 1])
        # First layer CNN
        W_conv1 = weight_variable([3, 3, 1, 32])            # Where are these variables 'weight_variable' and 'bias_variable' defined?
        B_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + B_conv1)
        h_conv1_bn=tf.contrib.layers.batch_norm(h_conv1)
        h_pool1 = maxpool2d(h_conv1_bn)
        #h_pool1=tf.contrib.layers.batch_norm(h_conv1)
        # Second layer CNN

        W_conv2 = weight_variable([3, 3, 32, 64])
        B_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + B_conv2)
        h_conv2_bn = tf.contrib.layers.batch_norm(h_conv2)
        h_pool2 = maxpool2d(h_conv2_bn)

        # Fully connected layer1
        W_fc1 = weight_variable([4 * 4 * 64, 1024])
        B_fc1 = bias_variable([1024])
        h_pool2_reshape = tf.reshape(h_pool2, shape=[-1, 4 * 4 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_reshape, W_fc1) + B_fc1)
        h_fc1_bn=tf.contrib.layers.batch_norm(h_fc1)

        # Dropout
        h_drop = tf.nn.dropout(h_fc1_bn, keep_prob)

        # Fully connected layer2
        W_fc2 = weight_variable([1024, 2])
        B_fc2 = bias_variable([2])
        y_conv = tf.matmul(h_drop, W_fc2)+B_fc2

        return y_conv


def privatizerdeconnn(data, noise_seed, batch_size, output_dim, noise_seed_dim, depth=[256, 128, 1], alpha=0.1):
    with tf.variable_scope("privatizer"):
        #input = noise_seed
        input=tf.cast(noise_seed, tf.float32)

        W_proj = weight_variable([noise_seed_dim, 4*4*depth[0]])

        B_proj = bias_variable([4*4*depth[0]])
        f_proj= tf.matmul(input, W_proj) + B_proj
        h_proj = tf.nn.relu(tf.contrib.layers.batch_norm(f_proj))
        print('h_proj', np.shape(h_proj))
        h_proj = tf.reshape(h_proj, [batch_size,4, 4, depth[0]])



        W_deconv1 = weight_variable([3, 3, depth[1], depth[0]])
        B_deconv1 = bias_variable([depth[1]])

        f_deconv1 = tf.nn.conv2d_transpose(h_proj, W_deconv1, output_shape=[batch_size, 8,8,depth[1]], strides=[1,2,2,1])

        deconv1 = tf.nn.bias_add(f_deconv1, B_deconv1)
        h_deconv1 = tf.nn.relu(tf.contrib.layers.batch_norm(deconv1))

        W_deconv2 = weight_variable([3, 3, depth[2], depth[1]])
        B_deconv2 = bias_variable([depth[2]])
        f_deconv2 = tf.nn.conv2d_transpose(h_deconv1, W_deconv2, output_shape=[batch_size,16,16,depth[2]], strides=[1,2,2,1])

        deconv2 = tf.nn.bias_add(f_deconv2, B_deconv2)

        out_noise= tf.reshape(tf.nn.tanh(deconv2), [-1, 16*16])
        out=tf.add(data, out_noise)
        return out


#Test method
def test_dec_cnn(Xtest, Y1test, Y2test, Size):
    y_dec = []
    err_vec = []
    xhat_test = []
    dist_test = []
    dist_total=0
    Ytest=Y2test
    Xtest=Xtest
    Ntest=Size
    batches=math.ceil(Ntest/mbsize)
    print('batches:',batches)
    residual=(batches)*mbsize-Ntest
    if residual>0:

        print('Residual', residual)
        residual_X, residual_Y1, residual_Y2 = nextbatch(residual, Xtest, Y1test, Y2test, Ntest)


        print('R_Y2', np.shape(residual_Y2))
        print('Y2', np.shape(Y2test))

        print('R_X',np.shape(residual_X))
        print('X', np.shape(Xtest))



        Ytest=np.vstack((Ytest, residual_Y2))

        print('Ytest_stack',np.shape(Ytest))


        Xtest=np.vstack((Xtest, residual_X))
        print('Xtest_stack', np.shape(Xtest))

    Ytest_onehot = tf.squeeze(tf.one_hot(indices=Ytest, depth=2), [1])
    Ytest_onehot = Ytest_onehot.eval()
    Ztest = np.random.normal(0.0, 1.0, [Ntest+residual, noise_seed_dim])
    print('Ztest_stack', np.shape(Ztest))

    for i in range(batches):
        lb=i*mbsize
        ub=(i+1)*mbsize
        print( np.shape(Xtest[lb:ub, :]))
        ytest = sess.run(y, feed_dict={X: Xtest[lb:ub, :], Y_onehot: Ytest_onehot[lb:ub, :], Z: Ztest[lb:ub, :], keep_prob: 1.0})

        xhattest = sess.run(X_hat, feed_dict={X: Xtest[lb:ub, :], Y_onehot: Ytest_onehot[lb:ub, :], Z: Ztest[lb:ub, :], keep_prob: 1.0})
        xhat_test.append(xhattest)
        dist= sess.run(distortion, feed_dict={X: Xtest[lb:ub, :], Z: Ztest[lb:ub, :]})
        dist_total +=dist

        ydec = np.argmax(ytest, axis=1)
        y_dec.append(ydec)
        ytrue = np.reshape(Ytest[lb:ub, :], [mbsize])
        err_test = np.abs(ytrue - ydec)
        err_vec.append(err_test)


    y_dec=np.reshape(y_dec, Ntest+residual)
    err_vec=np.reshape(err_vec, Ntest + residual)
    xhat_test=np.reshape(xhat_test, [Ntest + residual, -1])
    accuracy = 1 - np.mean(err_vec)
    dist_test=dist_total/(batches)
    print('Accuracy: %.3f, Distortion: %.3f' % (accuracy, dist))
    return accuracy, xhat_test, dist_test, y_dec


X_hat = privatizerdeconnn(X, Z, mbsize, Dim, noise_seed_dim, depth=[256, 128, 1], alpha=0.1)
y = adversarycnnbn(X_hat)


adversary_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_onehot, logits=y))
privatizer_nxent = -1.0 * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_onehot, logits=y))
distortion = tf.reduce_mean(tf.reduce_sum(tf.square(X - X_hat), axis=[1]))
privatizer_dist = penalty_rate * tf.square(tf.maximum(distortion - d - eps, 0))
privatizer_loss = privatizer_nxent + privatizer_dist

t_vars = tf.trainable_variables()
a_vars = [var for var in t_vars if var.name.startswith("adversary")]
p_vars = [var for var in t_vars if var.name.startswith("privatizer")]

a_train = tf.train.AdamOptimizer(learning_ratea).minimize(adversary_loss, var_list=a_vars)
p_train = tf.train.AdamOptimizer(learning_ratep).minimize(privatizer_loss, var_list=p_vars)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    prate_base = 1.1
    for iter in range(num_iter):
        start_time = time.time()

        batchX, batchY1, batchY2 = nextbatch(mbsize, Xtrain, Y1train, Y2train, Ntrain)
        batchY_onehot = tf.one_hot(indices=batchY2, depth=2)
        batchY_onehot = tf.squeeze(batchY_onehot, [1])
        batchZ = np.random.normal(0.0, 1.0, [mbsize, noise_seed_dim])


        _, A_loss_curr = sess.run([a_train, adversary_loss],
                                  feed_dict={X: batchX, Y_onehot: batchY_onehot.eval(), Z: batchZ, keep_prob: 0.5})

        if (iter%2000==0):
            prate=np.power(prate_base, 1+iter/2000)

        #if (iter%2==0):
        _, P_loss_curr = sess.run([p_train, privatizer_loss],
                                  feed_dict={X: batchX, Y_onehot: batchY_onehot.eval(), Z: batchZ,keep_prob: 0.5, penalty_rate:prate})
        if (iter % 10 == 0 & iter <= num_iter - 6):
            # print(sess.run(distortion, feed_dict={X: batchX, Y_onehot: batchY_onehot.eval(), Z: batchZ}))
            p_distortion = sess.run(distortion, feed_dict={X: batchX, Z: batchZ, penalty_rate:prate})
            print('iter:', iter)
            print('Adversary loss: ', A_loss_curr)
            print('Privatizer loss:', P_loss_curr)
            print('Distortion:', p_distortion)
            duration = time.time() - start_time
            print('loss: iter=%d, loss=%f, time=%.3f' % (iter, A_loss_curr, duration))
            a_loss.append(A_loss_curr)
            p_loss.append(P_loss_curr)
            p_dist.append(p_distortion)

    ###Privatize Trainning data
    acc_train, privatized_train, dist_train, decision_train = test_dec_cnn(Xtrain, Y1train, Y2train, Ntrain)
    ###Test CNN adversary
    acc_final, privatized_final, dist_final, decision_final = test_dec_cnn(Xtest, Y1test, Y2test, Ntest)

    result = {}
    result['acc_final'] = acc_final
    result['privatized_test'] = privatized_final
    result['dist_final'] = dist_final
    result['decision_final']=decision_final


    result['acc_train'] = acc_train
    result['privatized_train'] = privatized_train
    result['dist_train'] = dist_train
    result['decision_train']=decision_train

    scipy.io.savemat('GENKI_d1decon_batchedcnn_drop_out1', result)
    a_loss_arr = np.array(a_loss)
    p_loss_arr = np.array(p_loss)
    p_distortion_arr = np.array(p_dist)

    plt.figure(1)
    plt.plot(p_distortion_arr, label='Distortion')

    plt.title("Distortion")
    plt.legend()

    plt.figure(2)
    plt.plot(a_loss_arr, label='Adversary training loss')
    plt.plot(p_loss_arr, label='Privatizer training loss')

    plt.title("Cross Entropy")
    plt.legend()
    plt.show()
