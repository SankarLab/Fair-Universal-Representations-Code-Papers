import tensorflow as tf
import scipy.io
import numpy as np
import time
from nnfunctions import *
import matplotlib.pyplot as pt
# load the matrix with privatized and original data
mat = scipy.io.loadmat('privatizeddecon4.mat')
#dimension of each data sample
Dim = np.asscalar(mat['D'])
#K1: dimension of class of facial expression, K2 dimension of class of gender
K1 = np.asscalar(mat['K1'])
K2 = np.asscalar(mat['K2'])

Ntest = np.asscalar(mat['Ntest'])
Ntrain = np.asscalar(mat['Ntrain'])

Xtrain = mat['privatized_train']
#Xtest = mat['privatized_final']
Xtest= mat['Xtest']



#label of expression 1: smile, 0: not smile
Y1train = mat['y1_train']-1
Y1test = mat['y1_test']-1

#label of gender 1: female, 0: male
Y2train = mat['y2_train']-1
Y2test = mat['y2_test']-1

Ndata = Ntrain+Ntest
#Y1data = np.hstack((Y1train, Y1test))
#Y2data = np.hstack((Y2train, Y2test))

#Xdata = np.transpose(Xdata)
#Y1data = np.transpose(Y1data)
#Y2data = np.transpose(Y2data)


#Xtrain = np.transpose(Xtrain)
Y1train = np.transpose(Y1train)
Y2train = np.transpose(Y2train)


Xtest = np.transpose(Xtest)
Y1test = np.transpose(Y1test)
Y2test = np.transpose(Y2test)


X = tf.placeholder(tf.float32, shape = [None, Dim])
X_hat = tf.placeholder(tf.float32, shape = [None, Dim])
Y1 = tf.placeholder(tf.float32, shape = [None, K1])
Y2 = tf.placeholder(tf.float32, shape = [None, K2])
d = tf.placeholder(shape = [], dtype = tf.float32, name = "distortion_in")
keep_prob=tf.placeholder(tf.float32)

def nextbatch (batchsize, X, Y1, Y2, datalen):
    idx = np.arange(0, datalen)
    np.random.shuffle(idx)
    idx = idx[:batchsize]
    X_shuffled = [X[i, :] for i in idx]
    Y1_shuffled = [Y1[i, :] for i in idx]
    Y2_shuffled = [Y2[i, :] for i in idx]
    return np.asarray(X_shuffled), np.asarray(Y1_shuffled), np.asarray(Y2_shuffled)

def adversarycnnbn(inputs, keep_prob, reuse=False):
    with tf.variable_scope("adversary", reuse=reuse):
        x_image = tf.reshape(inputs, shape=[-1, 16, 16, 1])
        # First layer CNN
        W_conv1 = weight_variable([3, 3, 1, 32])
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

        #out = tf.nn.softmax(y_conv)

        return y_conv#out

def testAccuracy(Xtest, Ytest):
    Y1test_onehot = tf.squeeze(tf.one_hot(indices=Ytest, depth=2), [1])
    Y1test_onehot = Y1test_onehot.eval()

    ytest = sess.run(y, feed_dict={X: Xtest, keep_prob: 1.0})
    ydec = np.argmax(ytest, axis=1)
    ytrue = np.reshape(Ytest, [200])
    err_vec = np.abs(ytrue - ydec)
    print(err_vec)
    accuracy = 1 - np.mean(err_vec)
    print('Accuracy: %.3f' % accuracy)
    return ytrue, ydec, err_vec, ytest


mbsize = 128

W1 = tf.Variable(tf.zeros([Dim, K1]))
b1 = tf.Variable(tf.zeros([K1]))

y= adversarycnnbn(X, keep_prob)

loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y1, logits=y))

train_step = tf.train.AdamOptimizer(0.00001).minimize(loss) #1e-5 for nn, 1e-3 for logistic regression


sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for iter in range(3000):
    start_time = time.time()
    batchX, batchY1, batchY2 = nextbatch(mbsize, Xtrain, Y1train, Y2train, Ntrain)
    batchY1_onehot = tf.one_hot(indices = batchY1, depth = 2)
    batchY1_onehot = tf.squeeze(batchY1_onehot,[1])
    #batchY2_onehot = tf.one_hot(indices = batchY2, depth = 2)
    #batchY2_onehot = tf.squeeze(batchY2_onehot,[1])
    _, loss_A = sess.run([train_step, loss], feed_dict = {X: batchX, Y1: batchY1_onehot.eval(), keep_prob: 0.5})
    if iter%50 == 0:
        duration = time.time() - start_time
        print('loss: iter=%d, loss=%f, time=%.3f' % (iter, loss_A, duration))

#print('Distortion:', Distortion)
ytrue, ydec, err_vec, ybelief = testAccuracy(Xtest, Y1test)
result = {}
result['Y_true'] = ytrue
result['Y_decision'] = ydec
result['Error_vector'] = err_vec
result['Y_belief'] = ybelief
scipy.io.savemat('GENKI_d4_256_DeconUtility5', result)

