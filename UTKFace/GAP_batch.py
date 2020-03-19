import tensorflow as tf
import os
import scipy.io
import json
import numpy as np
import math
from PIL import Image
import time
import matplotlib.pyplot as plt
import matplotlib.image as img
from skimage.transform import resize
plt.switch_backend('agg')

path1 = '/home/mjpatel8/9-9/'

class Prep_UTK:
    def __init__(self, path):
        self.path = path
        self.classes = 2

        self.train_data = None
        self.train_labels = None
        self.train_labels_1 = None
        self.train_labels_2 = None
        self.train_labels_3 = None

        self.valid_data = None
        self.valid_labels = None
        self.valid_labels_1 = None
        self.valid_labels_2 = None
        self.valid_labels_3 = None

        self.test_data = None
        self.test_labels = None
        self.test_labels_1 = None
        self.test_labels_2 = None
        self.test_labels_3 = None

    def preprocess(self, encode=False):
        self.load_from_json()
        if encode:
            self.encode_labels()
        else:
            self.get_labels()
        return self.train_data, self.valid_data, self.test_data, [self.train_labels_1, self.train_labels_2, self.train_labels_3],\
                   [self.valid_labels_1, self.valid_labels_2, self.valid_labels_3], [self.test_labels_1, self.test_labels_2, self.test_labels_3]

    def img_to_arr(self, size):
        files = [file for file in os.listdir(self.path + 'train/')]
        size = size
        counter = 0
        arr = []
        labels = []
        for name in files:
            counter += 1
            # Resize and append to arr
            img1 = img.imread(self.path + 'train/' + name)
            #print(counter)
            img_resized = resize(img1, (size, size), anti_aliasing=True)
            arr.append(img_resized)
            # append labels
            inner = []
            lab_arr = name.split('_')
            inner.append(int(lab_arr[0]))
            inner.append(int(lab_arr[1]))
            inner.append(int(lab_arr[2]))
            labels.append(np.array(inner))

        return np.array(arr), np.array(labels)

    def load_from_json(self):
        try:
            data = scipy.io.loadmat(self.path + 'Train.mat')
            self.train_data = np.array(data['data'])
            self.train_labels = np.array(data['labels'])
            # with open(self.path + 'Train.mat') as file:

            data = scipy.io.loadmat(self.path + 'Valid.mat')
            self.valid_data = np.array(data['data'])
            self.valid_labels = np.array(data['labels'])

            # with open(self.path + 'test.json') as file:
            data = scipy.io.loadmat(self.path + 'Test.mat')
            self.test_data = np.array(data['data'])
            self.test_labels = np.array((data['labels']))
        except IOError:
            print('Incorrect File path')

    def get_labels(self):
        # get train labels
        self.train_labels_1 = self.train_labels[:, 0]
        self.train_labels_2 = self.train_labels[:, 1]
        self.train_labels_3 = self.train_labels[:, 2]

        self.valid_labels_1 = self.valid_labels[:, 0]
        self.valid_labels_2 = self.valid_labels[:, 1]
        self.valid_labels_3 = self.valid_labels[:, 2]

        # get test labels
        self.test_labels_1 = self.test_labels[:, 0]
        self.test_labels_2 = self.test_labels[:, 1]
        self.test_labels_3 = self.test_labels[:, 2]

prep = Prep_UTK(path1)
print('-----------> Loading Data')
x_train, x_valid, x_test, y_train, y_valid, y_test = prep.preprocess(encode=False)
print('-----------> Done Loading Data')
Ntrain = x_train.shape[0]
Xtrain = x_train
Xtest = x_test
print('Xtrain shape', Xtrain.shape)
print('Xtest shape', Xtest.shape)
print('Xvalid shape', x_valid.shape)

#define hyperparameters
mbsize = 128
epochs = 50
best_epoch = 0
best_acc = 100
learning_ratep = 0.0001
learning_ratea = 0.0002

d = 0.007  #distortion upper bound
print('---------------->')
print('Model Parameters')
print('Epochs', epochs)
print('Batch_Size', mbsize)
print('Distortion Upper Bound', d)
print('---------------->')

#define placeholders
X = tf.placeholder(tf.float32, shape=[None, 64,64,3])
batch_size = tf.placeholder(tf.int32)
Y = tf.placeholder(tf.float32, shape=[None, 2])
encoded_y = tf.placeholder(tf.float32, shape = [None, 2])
penalty_rate = tf.placeholder(tf.float32)
lrate_a = tf.placeholder(tf.float32)
lrate_p = tf.placeholder(tf.float32)


#define loss arrays
a_loss = []
p_loss = []
p_dist = []
v_dist = []
v_acc = []
save_arr = []

def adversary(inputs):
    with tf.variable_scope("adversary"):

        #rehshaping the input
        x_image = tf.reshape(inputs, shape=[-1, 64, 64, 3])

        # First layer CNN
        h_conv1 = tf.keras.layers.Conv2D(20, kernel_size=(3, 3), activation='relu', padding='same')(x_image)
        h_pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(h_conv1)

        # Second layer CNN
        h_conv2 = tf.keras.layers.Conv2D(20*2, kernel_size=(3, 3), activation='relu', padding='same')(h_pool1)
        h_pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(h_conv2)

        # Fully connected layer1
        h_pool2_reshape = tf.keras.layers.Flatten()(h_pool2)
        h_fc1 = tf.keras.layers.Dense(40, activation='relu')(h_pool2_reshape)

        # Fully connected layer2
        y = tf.keras.layers.Dense(2, activation='softmax')(h_fc1)

        return y

def privatizer(data, batch_size):
    with tf.variable_scope('privatizer'):
        # input_img = Input(shape=self.input_shape)
        x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(data)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        # at this point the representation is (8, 8, 64) i.e. 4096-dimensional
        encoded_f = tf.keras.layers.Flatten()(x)

        noise = tf.random_normal([batch_size,1000], mean = 0, stddev=1)
        concated = tf.concat([encoded_f, noise], 1)

        y = tf.keras.layers.Dense(4096, activation='relu')(concated)
        y = tf.keras.layers.Dense(4096, activation='relu')(y)
        noisy_output = tf.keras.layers.Reshape(target_shape=(8,8,64))(y)

        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(noisy_output)
        x = tf.keras.layers.UpSampling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.UpSampling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.UpSampling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        decoded = tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

        return decoded, encoded_f

def test(Xtest, Ytest, Size, type):
    Ytest_onehot = np.eye(2)[Ytest]
    ytest = []
    xhattest = []
    dist = []
    my_list = np.arange(Size)
    chunk_size = 128
    li = [my_list[i:i + chunk_size] for i in range(0, len(my_list), chunk_size)]

    for i in li:
        batchX = Xtest[i]
        label = Ytest_onehot[i]
        ytest.append(np.argmax(sess.run(y, feed_dict={X: batchX, encoded_y: label, batch_size: len(i)}), axis=1))
        xhattest.append(sess.run(X_hat, feed_dict={X: batchX, encoded_y: label, batch_size: len(i)}))
        dist.append(sess.run(distortion, feed_dict={X: batchX, batch_size: len(i)}))

    ytest = np.concatenate(ytest, axis=0)
    ytrue = np.reshape(Ytest, [Size])
    err_vec = np.abs(ytrue - ytest)
    accuracy = 1 - np.mean(err_vec)
    dist = np.array(dist).mean()
    print('Accuracy: %.7f, Distortion: %.7f' % (accuracy, dist))
    return accuracy, np.concatenate(xhattest, axis=0), dist, ytest

def test_valid(Xtest, Ytest, prate):
    Ytest_onehot = np.eye(2)[Ytest]
    ytest = []
    dist = []
    Size = Xtest.shape[0]
    my_list = np.arange(Size)
    chunk_size = 128
    li = [my_list[i:i + chunk_size] for i in range(0, len(my_list), chunk_size)]

    total_adv_loss = 0
    total_pvt_loss = 0
    for i in li:
        batchX = Xtest[i]
        label = Ytest_onehot[i]
        A_loss, P_loss = sess.run([adversary_loss, privatizer_loss], feed_dict={X: batchX, encoded_y: label, penalty_rate:prate, batch_size: len(i)})
        ytest.append(np.argmax(sess.run(y, feed_dict={X: batchX, encoded_y: label, batch_size: len(i)}), axis=1))
        dist.append(sess.run(distortion, feed_dict={X: batchX, batch_size: len(i)}))
        total_adv_loss += A_loss
        total_pvt_loss += P_loss

    ytest = np.concatenate(ytest, axis=0)
    ytrue = np.reshape(Ytest, [Size])
    err_vec = np.abs(ytrue - ytest)
    accuracy = 1 - np.mean(err_vec)
    dist = np.array(dist).mean()
    return accuracy, dist, total_adv_loss / len(li), total_pvt_loss / len(li)

def plot_examples():

    data = scipy.io.loadmat(path1 + 'data_new.mat')
    xhat = sess.run(X_hat, feed_dict={X:data['data'], batch_size: data['data'].shape[0]})
    preds = np.argmax(sess.run(y, feed_dict={X: data['data'], batch_size: data['data'].shape[0]}), axis=1)
    # print('Predictions', preds)
    print(xhat.shape)

    for i in range(xhat.shape[0]):
        plt.imsave('Examples/Img'+ str(i), xhat[i])
    img = []
    for i in range(data['data'].shape[0]):
        img.append(Image.open('Examples/Img'+ str(i) +'.png'))

    final_w = img[0].size[0] * data['data'].shape[0]
    final_h = img[0].size[1]

    result = Image.new('RGB', (final_w, final_h))
    for i in range(data['data'].shape[0]):
        result.paste(im=img[i], box=(i * img[0].size[0], 0))

    result.save('Examples/Combined.png')
    return preds

def model_summary():
    tf.contrib.slim.model_analyzer.analyze_vars(t_vars, print_info=True)

#create the graph
X_hat, x_com = privatizer(X, batch_size)
y = adversary(X_hat)

#defining Adversary Loss
# adversary_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=encoded_y, logits=y))       #change the last layer's activation to None if using this loss
adversary_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(encoded_y, y))

#defining Distortion
#distortion = tf.reduce_mean(tf.square(tf.subtract(X, X_hat)))      #to see the impact of high distortion value
distortion = tf.reduce_mean(tf.losses.mean_squared_error(X, X_hat))
#distortion = tf.reduce_mean(tf.losses.absolute_difference(X, X_hat))   #L1 losss

#Enforcing distortion constraint
dist_margin = tf.square(tf.maximum(distortion - d, 0))

#defining Privatizer Loss
privatizer_loss = (-1.0 * adversary_loss) + (penalty_rate * dist_margin)

# define trainable variables
t_vars = tf.trainable_variables()
a_vars = [var for var in t_vars if var.name.startswith("adversary")]
p_vars = [var for var in t_vars if var.name.startswith("privatizer")]

#generate model summary
model_summary()

#defining optimizers
a_train = tf.train.AdamOptimizer(lrate_a).minimize(adversary_loss, var_list=a_vars)
p_train = tf.train.AdamOptimizer(lrate_p).minimize(privatizer_loss, var_list=p_vars)

#saver to save the models
saver = tf.train.Saver(max_to_keep=10)

saver_adv = tf.train.Saver(var_list=a_vars)
saver_pvt = tf.train.Saver(var_list=p_vars)
#training Loop
res = open('GAP_Log.txt', 'w')
with tf.Session() as sess:

    # base prate
    prate = 10000
    sess.run(tf.global_variables_initializer())
    saver_pvt.restore(sess, 'Saved_Models/model_100.ckpt')
    saver_adv.restore(sess, 'Saved_Models/model_50_1.ckpt')
    #Epochal training

    acc_train, privatized_train, dist_train, decision_train = test(Xtrain, y_train[1], Xtrain.shape[0], 'Init')

    for epoch in range(epochs + 1):

        train_list = np.random.permutation(np.arange(x_train.shape[0]))

        chunk_size = mbsize
        batch = [train_list[i:i + chunk_size] for i in range(0, len(train_list), chunk_size)]
        start_time = time.time()

        #Iterate over all batches for given epoch
        for index, i in enumerate(batch):
            batchX = Xtrain[i]
            label = y_train[1][i]
            batchY_onehot = np.eye(2)[label]
            #train adversary more than the privatizer 20 times
            train_adv = 20
            for iny in range(train_adv):
                _ = sess.run([a_train],
                                     feed_dict={X: batchX, encoded_y: batchY_onehot, lrate_a: learning_ratea, batch_size: len(i)})
            # _, A_loss_curr = sess.run([a_train, adversary_loss], feed_dict={X: batchX, encoded_y: batchY_onehot, lrate_a: learning_ratea, batch_size: len(i)})
            _ = sess.run([p_train], feed_dict={X: batchX, encoded_y: batchY_onehot, penalty_rate:prate, lrate_p: learning_ratep, batch_size: len(i)})
        train_acc, train_dist, train_adv_loss, train_pvt_loss = test_valid(x_train, y_train[1], prate)
        valid_acc, valid_dist, valid_adv_loss, valid_pvt_loss = test_valid(x_valid, y_valid[1], prate)
        a_loss.append(train_adv_loss)
        p_loss.append(train_pvt_loss)
        p_dist.append(train_dist)

        v_dist.append(valid_dist)
        v_acc.append(valid_acc)
        duration = time.time() - start_time
        print(
            'Epoch=%d, Adversary_loss=%f, Privatizer_loss=%f, Training Distortion=%f, Validation Distortion=%f, Training Acc=%f, Validation Acc=%f, Training Time=%.3f' % (
            epoch, train_adv_loss, train_pvt_loss, train_dist, valid_dist, train_acc, valid_acc, duration))
        res.write('Epoch=%d, Adversary_loss=%f, Privatizer_loss=%f, Training Distortion=%f, Validation Distortion=%f, Training Acc=%f, Validation Acc=%f, Training Time=%.3f \n' % (
            epoch, train_adv_loss, train_pvt_loss, train_dist, valid_dist, train_acc, valid_acc, duration))
        if train_acc > 0.90 and valid_acc < best_acc:
            best_epoch = epoch
            best_acc = valid_acc
            saved_path = saver.save(sess, 'Models/GAP_Model_' + str(epoch) + '.ckpt')
        if epoch % 6 == 0:
            learning_ratea = learning_ratea * 0.90
            learning_ratep = learning_ratep * 0.80
            print('Adversary Learning Rate changed to %f' % (learning_ratea))
            print('Privatizer Learning Rate changed to %f' % (learning_ratep))
        if train_dist > d:
            temp = ((d - train_dist) ** 2) * prate
            while temp < 0.1:
                prate = prate * 10
                temp = ((d - train_dist) ** 2) * prate
            print('Penalty Rate Changed to %f' % (prate))
            res.write('Penalty Rate Changed to %f \n' % (prate))

    saved_path = saver.save(sess, 'Models/GAP_Model_' + str(epochs) + '.ckpt', write_meta_graph=False)
    print('Model saved at ', saved_path)
    if best_epoch > 0:
        saver.restore(sess, 'Models/GAP_Model_'+ str(best_epoch) + '.ckpt')
        print('Model %d restored' %(best_epoch))
        res.write('GAP_Model %d restored \n' %(best_epoch))

    # save losses
    losses = {}
    losses['Adv_loss'] = a_loss
    losses['Pvt_Loss'] = p_loss
    losses['Train_Dist'] = p_dist
    losses['Valid_Dist'] = v_dist
    losses['Valid_Acc'] = v_acc
    scipy.io.savemat('Models/GAP_Losses.mat', losses)

    # plot the losses
    a_loss_arr = np.array(a_loss)
    p_loss_arr = np.array(p_loss)
    p_distortion_arr = np.array(p_dist)

    plt.plot(p_distortion_arr, label='Distortion Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Distortion')
    plt.title("Distortion Curve")
    plt.legend()
    plt.savefig('Models/GAP_Model_Distortion.png')
    plt.clf()

    plt.plot(a_loss_arr, label='Adversary training loss')
    plt.plot(p_loss_arr, label='Privatizer training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title("Loss Curve")
    plt.legend()
    plt.savefig('Models/GAP_Model_Loss.png')
    plt.clf()

    plt.plot(np.log(a_loss_arr), label='Adversary Log Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Log Loss')
    plt.title('Adversary Log Loss')
    plt.legend()
    plt.savefig('Models/GAP_Adv_LogLoss.png')
    plt.clf()

    plt.plot(v_acc, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Valid_Acc')
    plt.legend()
    plt.savefig('Models/GAP_Validation.png')
    plt.clf()
    res.write('Predictions: ')
    preds = plot_examples()
    for item in preds:
        res.write('%d,' % item )
    res.write('\n')
    #Privatize training data and compute accuracy
    acc_train, privatized_train, dist_train, decision_train = test(Xtrain, y_train[1], Xtrain.shape[0], 'train')
    res.write('Training Accuracy: %.7f, Training Distortion: %.7f \n' % (acc_train, dist_train))

    # Privatize valid data and compute accuracy
    acc_valid, privatized_valid, dist_valid, decision_valid = test(x_valid, y_valid[1], x_valid.shape[0], 'valid')
    res.write('Validation Accuracy: %.7f, Validation Distortion: %.7f \n' % (acc_valid, dist_valid))

    #Privatize test data and compute accuracy
    acc_test, privatized_test, dist_test, decision_test= test(Xtest, y_test[1], Xtest.shape[0], 'test')
    res.write('Testing Accuracy: %.7f, Testing Distortion: %.7f \n' % (acc_test, dist_test))


    #save results to mat file
    result = {}
    result['acc_test'] = acc_test
    result['privatized_test'] = privatized_test
    result['dist_test'] = dist_test
    result['decision_test'] = decision_test
    result['y_test_age'] = y_test[0]
    result['y_test_gender'] = y_test[1]
    result['y_test_race'] = y_test[2]

    result['acc_train'] = acc_train
    result['privatized_train'] = privatized_train
    result['dist_train'] = dist_train
    result['decision_train'] = decision_train
    result['y_train_age'] = y_train[0]
    result['y_train_gender'] = y_train[1]
    result['y_train_race'] = y_train[2]

    result['acc_valid'] = acc_valid
    result['privatized_valid'] = privatized_valid
    result['dist_valid'] = dist_valid
    result['decision_valid'] = decision_valid
    result['y_valid_age'] = y_valid[0]
    result['y_valid_gender'] = y_valid[1]
    result['y_valid_race'] = y_valid[2]

    scipy.io.savemat('Data_Noisy', result)

res.close()

import Gender_Keras
import Race_Keras_VGG
