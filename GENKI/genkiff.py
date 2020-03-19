# Feed forward based privatizer
import tensorflow as tf
import scipy.io
import numpy as np
import time
from nnfunctions import *                           # directly use the name of functions and variables defined in nnfunctions.py without nnfunctions.**
import matplotlib.pyplot as plt

##**************************Loading data from 'genki.mat' *************************
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
Xdata = np.hstack((Xtrain, Xtest))    # add columns of Xtest to after the columns of Xtest
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

print('Shape of X', np.shape(Xdata))
print('Shape of Y1',np.shape(Y1data))
print('Shape of Y2',np.shape(Y2data))

mbsize = 200                        # batch size
num_iter = 20000                    # number of epochs= num_iter/(# training data/mbsize )
tradeoff = 1                        # ?

learning_ratep = 0.0002             # Learning rate for privatizer
learning_ratea = 0.0002             # Learning rate for adversary
eps =0                              # Not updated by the algorithm??
noise_seed_dim = 100                # the size of the noise 1 times noise_seed_dim
d = 1                               # D: Upper bound of the accepted distortion

X = tf.placeholder(tf.float32, shape=[None, Dim])
Y = tf.placeholder(tf.float32, shape=[None, K1])
Y_onehot = tf.placeholder(tf.float32, shape=[None, K1])
Z = tf.placeholder(tf.float32, shape=[None, noise_seed_dim], name='Z')
keep_prob = tf.placeholder(tf.float32)     # the parameter for dropout
penalty_rate = tf.placeholder(tf.float32)  #


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
def adversarycnnbn(inputs, keep_prob, reuse=False):
    with tf.variable_scope("adversary", reuse=reuse):
        x_image = tf.reshape(inputs, shape=[-1, 16, 16, 1])
        # First layer CNN
        W_conv1 = weight_variable([3, 3, 1, 32])    # size of filter: 3 times 3; input channel: 1; output channel: 32
        B_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + B_conv1)
        h_conv1_bn=tf.contrib.layers.batch_norm(h_conv1)
        h_pool1 = maxpool2d(h_conv1_bn)             # reduce the image size to 8 times 8
        # Second layer CNN

        W_conv2 = weight_variable([3, 3, 32, 64])
        B_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + B_conv2)
        h_conv2_bn = tf.contrib.layers.batch_norm(h_conv2)
        h_pool2 = maxpool2d(h_conv2_bn)             # reduce the image size to 4 times 4

        # Fully connected layer1
        W_fc1 = weight_variable([4 * 4 * 64, 1024])
        B_fc1 = bias_variable([1024])
        h_pool2_reshape = tf.reshape(h_pool2, shape=[-1, 4 * 4 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_reshape, W_fc1) + B_fc1)
        h_fc1_bn=tf.contrib.layers.batch_norm(h_fc1)

        # Dropout
        h_drop = tf.nn.dropout(h_fc1_bn, keep_prob)      # randomly set some values to be 0 with the probability indicated by 1-keep_prob

        # Fully connected layer2
        W_fc2 = weight_variable([1024, 2])
        B_fc2 = bias_variable([2])
        y_conv = tf.matmul(h_drop, W_fc2)+B_fc2


        return y_conv

#Feedforward privatizer structure
def privatizernn(data, noise_seed, structure=[256, 256, 256], alpha=0.1):
    with tf.variable_scope("privatizer"):
        input = tf.concat(values=[data, noise_seed], axis=1)
        fc1 = fc_bn_leakyRelu(input, structure[0], alpha=alpha)
        fc2 = fc_bn_leakyRelu(fc1, structure[1], alpha=alpha)
        fc3 = fc_bn_leakyRelu(fc2, structure[2], alpha=alpha)
        x_hat = tf.layers.dense(fc3, Dim, activation=None)         # tf.layers.dense: full connected layer
        return x_hat
# remark: noise_seed is the Z which can be different for each sample

#Test method
def test_mix_cnn(Xtest, Ytest, Size):
    print('Xtest:', Xtest)
    Ytest_onehot = tf.squeeze(tf.one_hot(indices=Ytest, depth=2), [1])  # size of output of tf.one_hot: batchsize*1*depth
    Ytest_onehot = Ytest_onehot.eval()   # to use it in session
    # Ytest_oneshot is not needed here.
    Ztest = np.random.normal(0.0, 1.0, [Size, noise_seed_dim])          # generate the noise with the dimension as batchsize times 100
    ytest = sess.run(y, feed_dict={X: Xtest, Y_onehot: Ytest_onehot, Z: Ztest, keep_prob: 1.0}) ## where the term 'y' is defined?

    xhattest = sess.run(X_hat, feed_dict={X: Xtest, Y_onehot: Ytest_onehot, Z: Ztest, keep_prob: 1.0})
    ydec = np.argmax(ytest, axis=1)
    ytrue = np.reshape(Ytest, [Size])
    err_vec = np.abs(ytrue - ydec)
    print(ydec)
    accuracy = 1 - np.mean(err_vec)
    dist = sess.run(distortion, feed_dict={X: batchX, Z: batchZ})       # 不能feed中间变量 X_hat 是 X 和 Z 产生的
    print('Accuracy: %.3f, Distortion: %.3f' % (accuracy, dist))
    return accuracy, xhattest, dist, ydec


X_hat = privatizernn(X, Z)

y = adversarycnnbn(X_hat, keep_prob)


## Two objective functions in the GAP-training  phrase
adversary_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_onehot, logits=y)) # tf.reduce_mean: return the mean

privatizer_nxent = -1.0 * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_onehot, logits=y))

distortion = tf.reduce_mean(tf.reduce_sum(tf.square(X - X_hat), axis=[1]))

#Enforcing distortion constraint
dist_margin=tf.square(tf.maximum(distortion - d - eps, 0))
privatizer_dist = penalty_rate * dist_margin

# privatizer_dist=0
privatizer_loss = privatizer_nxent + privatizer_dist

t_vars = tf.trainable_variables()      #  a list of all variables
a_vars = [var for var in t_vars if var.name.startswith("adversary")]
p_vars = [var for var in t_vars if var.name.startswith("privatizer")]

##
a_train = tf.train.AdamOptimizer(learning_ratea).minimize(adversary_loss, var_list=a_vars)
p_train = tf.train.AdamOptimizer(learning_ratep).minimize(privatizer_loss, var_list=p_vars)

with tf.Session() as sess:
    prate_base=1.1      # initial value of the penalty rate
    sess.run(tf.global_variables_initializer())
    for iter in range(num_iter):
        start_time = time.time()
        #Minibatch sampling
        batchX, batchY1, batchY2 = nextbatch(mbsize, Xtrain, Y1train, Y2train, Ntrain)
        batchY_onehot = tf.one_hot(indices=batchY2, depth=2)
        batchY_onehot = tf.squeeze(batchY_onehot, [1])
        batchZ = np.random.normal(0.0, 1.0, [mbsize, noise_seed_dim])

        #Training adversary
        _, A_loss_curr = sess.run([a_train, adversary_loss],
                                  feed_dict={X: batchX, Y_onehot: batchY_onehot.eval(), Z: batchZ, keep_prob: 0.5})

        #Adjusting penalty rate
        if (iter%2000==0):
            prate=np.power(prate_base, 1+iter/2000)

        #Training privatizer
        _, P_loss_curr = sess.run([p_train, privatizer_loss],
                                  feed_dict={X: batchX, Y_onehot: batchY_onehot.eval(), Z: batchZ, keep_prob: 0.5, penalty_rate:prate})
        if (iter % 100 == 0 & iter <= num_iter - 6):
            # print(sess.run(distortion, feed_dict={X: batchX, Y_onehot: batchY_onehot.eval(), Z: batchZ}))
            p_distortion = sess.run(distortion, feed_dict={X: batchX, Z: batchZ})
            print('iter:', iter)
            print('prate:',prate)
            print('Adversary loss: ', A_loss_curr)
            print('Privatizer loss:', P_loss_curr)
            print('Distortion:', p_distortion)
            print('privatizer_dist:', sess.run(privatizer_dist, feed_dict={X: batchX, Z: batchZ, penalty_rate:prate}))
            duration = time.time() - start_time
            print('loss: iter=%d, loss=%f, time=%.3f' % (iter, A_loss_curr, duration))
            a_loss.append(A_loss_curr)
            p_loss.append(P_loss_curr)
            p_dist.append(p_distortion)

    #Privatize training data
    acc_train, privatized_train, dist_train, decision_train = test_mix_cnn(Xtrain, Y2train, Ntrain)

    ###Test CNN adversary
    acc_final, privatized_final, dist_final, decision_final = test_mix_cnn(Xtest, Y2test, Ntest)
    result = {}
    result['acc_final'] = acc_final
    result['privatized_final'] = privatized_final
    result['dist_final'] = dist_final
    result['decision_final']=decision_final

    result['acc_train'] = acc_train
    result['privatized_train'] = privatized_train
    result['dist_train'] = dist_train
    result['decision_train'] = decision_train

    scipy.io.savemat('GENKI_d1_256_feedforward4', result)
    a_loss_arr = np.array(a_loss)
    p_loss_arr = np.array(p_loss)
    p_distortion_arr = np.array(p_dist)

    plt.figure(1)
    plt.plot(p_distortion_arr, label='Distortion')

    plt.title("Distortion")
    plt.legend()
    plt.show()

    plt.figure(2)
    plt.plot(a_loss_arr, label='Adversary training loss')
    plt.plot(p_loss_arr, label='Privatizer training loss')

    plt.title("Cross Entropy")
    plt.legend()
    plt.show()
