import numpy as np
import tensorflow as tf
import math
import time
import scipy.io
from nnfunctions import *
import matplotlib.pyplot as plt

data = np.loadtxt(open("HAR_privatized_d1_5_full.csv", "rb"), dtype= float, delimiter=",", skiprows=1)



# Y1 action, Y2 identity
N_train = 8000
N_validate = 1000
X_data_train = data[0:N_train, 0:561]

Iden_data_train = data[0:N_train, 561]

Act_data_train = data[0:N_train, 562]



Act_data_train = np.reshape(Act_data_train, (Act_data_train.shape[0], 1))
Iden_data_train = np.reshape(Iden_data_train, (Iden_data_train.shape[0], 1))

Xtrain = X_data_train[0: N_train, :]

Y1train = Act_data_train[0: N_train, :]
Y2train = Iden_data_train[0: N_train, :]

Xvalidate = X_data_train[N_train - N_validate: N_train, :]
Y1validate = Act_data_train[N_train - N_validate: N_train, :]
Y2validate = Iden_data_train[N_train - N_validate: N_train, :]




X_data_test = data[N_train:, 0:561]
N_test = X_data_test.shape[0]
Iden_data_test = data[N_train:, 561]

Act_data_test = data[N_train:, 562]



Act_data_test = np.reshape(Act_data_test, (Act_data_test.shape[0], 1))
Iden_data_test = np.reshape(Iden_data_test, (Iden_data_test.shape[0], 1))
Xtest = X_data_test
Y1test = Act_data_test
Y2test = Iden_data_test
#print(np.max(Xtest))

epoch = 450
mbsize = 128
noise_seed_dim =100
num_iter = int(epoch * (N_train)/mbsize)
print(num_iter)


N_act = 6
N_iden = 30

N_feature = X_data_train.shape[1]
N_all = N_train + N_test


a_loss = []
p_loss = []
p_dist = []


learning_ratea = 0.00005
noise_seed_dim =100

X = tf.placeholder(dtype = tf.float32, shape = [None, 561])
Y_act = tf.placeholder(dtype = tf.float32, shape = [None, N_act])
Y_iden = tf.placeholder(dtype = tf.float32, shape = [None, N_iden])
Y_onehot = tf.placeholder(tf.float32, shape=[None, N_iden])
Z = tf.placeholder(tf.float32, shape=[None, noise_seed_dim], name='Z')
keep_prob = tf.placeholder(tf.float32)
penalty_rate = tf.placeholder(tf.float32)



def minibatch(X, action, identity,mbsize, N_sample):
    idx = np.arange(N_sample)
    np.random.shuffle(idx)
    idx = idx[0:mbsize]
    X_mb = [X[i, :] for i in idx]
    Y_act = [action[i, :] for i in idx]
    Y_iden = [identity[i, :] for i in idx]
    return np.asarray(X_mb), np.asarray(Y_act), np.asarray(Y_iden)



def privatizernn(data, noise_seed, structure=[512, 512, 512], alpha=0.1, keep_prob = 1.0):
    with tf.variable_scope("privatizer"):
        input = tf.concat(values=[data, noise_seed], axis=1)
        fc1 = fc_bn_leakyRelu(input, structure[0], alpha=alpha, keep_prob = keep_prob)

        fc2 = fc_bn_leakyRelu(fc1, structure[1], alpha=alpha, keep_prob = keep_prob)
        fc3 = fc_bn_leakyRelu(fc2, structure[2], alpha=alpha, keep_prob = keep_prob)
        x_hat = tf.layers.dense(fc3, data.shape[1], activation=None)
        return x_hat

# def adversarynn(data, num_out, structure = [512, 512, 512], alpha = 0.1, keep_prob = 1.0):
#     with tf.variable_scope("adversary"):
#         fc1_a = fc_bn_leakyRelu(data, structure[0], alpha = alpha, keep_prob = keep_prob)
#         fc2_a = fc_bn_leakyRelu(fc1_a, structure[1], alpha=alpha, keep_prob = keep_prob)
#         fc3_a = fc_bn_leakyRelu(fc2_a, structure[2], alpha=alpha, keep_prob = keep_prob)
#         h_hat = tf.layers.dense(fc3_a, num_out, activation=None)
#         #y_hat = tf.nn.softmax(h_hat)
#         return h_hat

def adversarynn(data, num_out, structure = [512, 512, 256, 128], alpha = 0.1, keep_prob = 1.0):
    with tf.variable_scope("adversary"):
        fc1_a = fc_bn_leakyRelu(data, structure[0], alpha = alpha, keep_prob = keep_prob)
        fc2_a = fc_bn_leakyRelu(fc1_a, structure[1], alpha=alpha, keep_prob = keep_prob)
        fc3_a = fc_bn_leakyRelu(fc2_a, structure[2], alpha=alpha, keep_prob = keep_prob)
        fc4_a = fc_bn_leakyRelu(fc3_a, structure[3], alpha=alpha, keep_prob = keep_prob)
        h_hat = tf.layers.dense(fc4_a, num_out, activation=None)
        #y_hat = tf.nn.softmax(h_hat)
        return h_hat

def test_ff_nn(Xtest, Ytest, Size):
    #print('Xtest:', Xtest)
    Ytest_onehot = tf.squeeze(tf.one_hot(indices=Ytest, depth=N_iden), [1])
    Ytest_onehot = Ytest_onehot.eval()
    Ztest = np.random.normal(0.0, 1.0, [Size, noise_seed_dim])
    ytest = sess.run(y, feed_dict={X: Xtest, Y_onehot: Ytest_onehot, Z: Ztest, keep_prob: 1.0})

    #xhattest = sess.run(X_hat, feed_dict={X: Xtest, Y_onehot: Ytest_onehot, Z: Ztest, keep_prob: 1.0})
    ydec = np.argmax(ytest, axis=1)
    ytrue = np.reshape(Ytest, [Size])
    err_rate = np.mean(ytrue != ydec)
    print(ydec)
    print(ytrue)
    #print(err_rate)
    accuracy = 1 - err_rate
    #dist = sess.run(distortion, feed_dict={X: batchX, Z: batchZ})
    #print('Accuracy: %.3f, Distortion: %.3f' % (accuracy, dist))
    return accuracy#, xhattest, dist, ydec

y = adversarynn(X, N_iden, keep_prob = keep_prob)


adversary_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_onehot, logits=y))


t_vars = tf.trainable_variables()
a_vars = [var for var in t_vars if var.name.startswith("adversary")]
p_vars = [var for var in t_vars if var.name.startswith("privatizer")]

a_train = tf.train.AdamOptimizer(learning_ratea).minimize(adversary_loss, var_list=a_vars)

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    for iter in range (num_iter):
        start_time = time.time()
        batchX, batchY1, batchY2 = minibatch(Xtrain, Y1train, Y2train, mbsize, N_train)
        batchY_onehot = tf.one_hot(indices = batchY2, depth = N_iden)
        batchY_onehot = tf.squeeze(batchY_onehot, [1])
        batchZ = np.random.normal(0.0, 1.0, [mbsize, noise_seed_dim])

        #Training adversary
        _, A_loss_curr = sess.run([a_train, adversary_loss],
                                  feed_dict={X: batchX, Y_onehot: batchY_onehot.eval(), Z: batchZ, keep_prob: 0.5})


        if(iter % 100 == 0):
            #print('iter:', iter)
            #print('prate:', prate)
            #print('Adversary loss: ', A_loss_curr)
            #print('Privatizer loss:', P_loss_curr)
            #print('Distortion:', p_distortion)
            #print('privatizer_dist:', sess.run(privatizer_dist, feed_dict={X: batchX, Z: batchZ, penalty_rate: prate}))
            duration = time.time() - start_time
            print('loss: iter=%d, loss=%f, time=%.3f' % (iter, A_loss_curr, duration))
            a_loss.append(A_loss_curr)
            #p_loss.append(P_loss_curr)
            #p_dist.append(p_distortion)

    acc_final= test_ff_nn(Xtest, Y2test, N_test)
    # Save the variables to disk.
    save_path = saver.save(sess, "/tmp/HAR/Iden/model_privatized1_5.ckpt")
    print("Model saved in path: %s" % save_path)
    print(acc_final)
    result = {}
    result['acc_final'] = acc_final

    a_loss_arr = np.array(a_loss)
    scipy.io.savemat('HAR_1_5iden', result)

    plt.figure(1)

    plt.plot(a_loss_arr, label='Adversary training loss')
    #plt.plot(p_loss_arr, label='Privatizer training loss')

    plt.title("Cross Entropy")
    plt.legend()
    plt.show()
