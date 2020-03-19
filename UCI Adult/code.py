#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import math
import time
import scipy.io
from nnfunctions import *
import matplotlib.pyplot as plt
import scipy.io as sio 
import sys
import csv
import os.path


# In[ ]:





# In[2]:


def read_mat_file(filename):
    file_content = sio.loadmat(filename)
    X_train = file_content['X_train']
    Y_train = file_content['Y_train']
    Y_train_sal = file_content['Y_train_sal']

    return X_train, Y_train, Y_train_sal


# In[3]:


def male_female_data_seperate(X_test, Y_test, Y_salary):
    male_x = []
    male_s = []
    female_s = []
    female_x = []

    for i in range(Y_test.shape[0]):
        if Y_test[i,0] == 1:
            female_x.append(X_test[i,:])
            female_s.append(Y_salary[i])
        else:
            male_x.append(X_test[i, :])
            male_s.append(Y_salary[i])

    male_x = np.array(male_x)
    female_x = np.array(female_x)
    male_s = np.array(male_s)
    female_s = np.array(female_s)

    return male_x, female_x


# In[4]:


def save_the_train_data_set(X_hat_data_train, X_hat_data_test, Y_data_train, Y_data_test, d1, d2):
    sio.savemat("Representation/UCI_Adult_Generated - " + str(d1) + "_" + str(d2) + ".mat", {'X_train': X_hat_data_train , 'Y_train': Y_data_train, 'X_test': X_hat_data_test, 'Y_test' : Y_data_test})


# In[5]:


X_data_train, Y_data_train, Y_data_sal_train = read_mat_file("../UCI_Adult_Train_data_for_GAP_Complete.mat")  # X_train : 30162 x 109
X_data_test, Y_data_test, Y_data_sal_test = read_mat_file("../UCI_Adult_Test_data_for_GAP_Complete.mat")  # X_train : 30162 x 109

N_train = X_data_train.shape[0]
N_test = X_data_test.shape[0]

mbsize = 2048
epoch = int(sys.argv[5]) #500
learning_ratea = 0.001 # I changed it from ` to o.001
learning_ratep = 0.001


# In[6]:


d1 = float(sys.argv[1]) # This one is for the categorical features 0.1#
d2 = float(sys.argv[2]) # This one is for the continuous features 0.04#
countinous_bound = float(sys.argv[3]) # This is to set the penalty rate! 20#
lambda_ = float(sys.argv[4]) #0.5#


# In[7]:


overall_distortion_value = d1 + d2

# print ("#Distortion Upper Bound")
# print (d1)
# print (d2)


# In[8]:


import datetime
log_file_name = str(d1) + "_" + str(d2) + "_" + str(countinous_bound) +"_" + str(lambda_) +"_" + str(epoch) + "_" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+"_log.csv"
def append_record(record):
    with open(log_file_name, 'a') as f:
        record_str = ""
        for r in record:
            record_str = record_str + "," + str(r)        
        f.write(record_str[1:])
        f.write(os.linesep)


# In[9]:


log_file_name


# In[10]:


append_record(["epoch,time_taken,privatizer_loss,sal_classifier_loss,adversary_loss,distortion,age_distortion,work_distortion,education_distortion,marital_distortion,occupation_distortion,race_distortion,native_country_distortion,relationship_distortion,continous_distortion,diff_train, acc_train, accuracy_sal_train, cat_d, con_d, age_d, work_d, edu_d, mar_d, occ_d, race_d, native_d, relationship_d,diff_test ,acc_test, accuracy_sal_test, cat_d_t, con_d_t, age_d_t, work_d_t, edu_d_t, mar_d_t, occ_d_t, race_d_t, native_d_t, relationship_d_t"])


# In[11]:


# append_record([5,6,7])


# In[12]:


# with open(log_file_name, 'a') as f:
# #         json.dump(record, f)
#         f.write("aa")
#         f.write(os.linesep)


# In[13]:


noise_seed_dim = X_data_train.shape[1]
num_iter = int(N_train/mbsize)

# print("Total interations:", num_iter)

#N_act = 10
N_label = Y_data_train.shape[1]
N_feature = X_data_train.shape[1]
N_all = N_train + N_test

a_loss = []
p_loss = []
p_dist = []


# In[14]:


X = tf.placeholder(dtype = tf.float32, shape = [None, N_feature])
Y_onehot = tf.placeholder(tf.float32, shape=[None, N_label])
sal_onehot = tf.placeholder(tf.float32, shape=[None, N_label])

Z = tf.placeholder(dtype = tf.float32, shape=[None, noise_seed_dim], name='Z')
keep_prob = tf.placeholder(tf.float32)
penalty_rate = tf.placeholder(tf.float32)
p_keep_prob = tf.placeholder(tf.float32)
a_keep_prob = tf.placeholder(tf.float32)


# In[15]:


def shuffle_data(X_data_train, Y_data_train, Y_data_sal_train):
    idx = np.arange(X_data_train.shape[0])
    np.random.shuffle(idx)
    x_data = [X_data_train[i,:] for i in idx]
    y_data = [Y_data_train[i,:] for i in idx]
    sal_data = [Y_data_sal_train[i,:] for i in idx]

    return np.asarray(x_data) , np.asarray(y_data), np.asarray(sal_data)

def adversarynn(data, num_out, structure = [10, 5], alpha = learning_ratea, keep_prob = 1.0): # learning rate is changed from 0.1 to 0.01.
    with tf.variable_scope("adversary"):
        fc1_a = fc_bn_leakyRelu(data, structure[0], alpha = alpha, keep_prob = keep_prob)
        fc2_a = fc_bn_leakyRelu(fc1_a, structure[1], alpha=alpha, keep_prob = keep_prob)
        h_hat = tf.layers.dense(fc2_a, num_out, activation=None)
        return h_hat

def sal_classifier(data, num_out, structure = [10, 5], alpha = learning_ratea, keep_prob = 1.0): # learning rate is changed from 0.1 to 0.01.
    with tf.variable_scope("sal_clssifier"):
        fc1_a = fc_bn_leakyRelu(data, structure[0], alpha = alpha, keep_prob = keep_prob)
        fc2_a = fc_bn_leakyRelu(fc1_a, structure[1], alpha=alpha, keep_prob = keep_prob)
        s_hat = tf.layers.dense(fc2_a, num_out, activation=None)
        return s_hat

# def privatizernn(data, noise_seed, structure=[20,15], alpha=learning_ratep, keep_prob = 1.0):
#     with tf.variable_scope("privatizer"):
#         input = tf.concat(values=[data, noise_seed], axis=  1)
#         fc1 = fc_bn_leakyRelu(input, structure[0], alpha=alpha, keep_prob = keep_prob)
#         #fc2 = fc_bn_leakyRelu(fc1, structure[1], alpha=alpha, keep_prob = keep_prob)
#         #fc3 = fc_bn_leakyRelu(fc2, structure[2], alpha=alpha, keep_prob = keep_prob)
#         x_hat = tf.layers.dense(fc1, data.shape[1] - 2, activation=None) # -2, Remove the Gender representation !

#         return x_hat

def privatizernn(data, noise_seed, structure=[170,130,113], alpha=learning_ratep, keep_prob = 1.0):
    with tf.variable_scope("privatizer"):
        input = tf.concat(values=[data, noise_seed], axis=  1)
        fc1 = fc_bn_leakyRelu(input, structure[0], alpha=alpha, keep_prob = keep_prob)
        fc2 = fc_bn_leakyRelu(fc1, structure[1], alpha=alpha, keep_prob = keep_prob)
        fc3 = fc_bn_leakyRelu(fc2, structure[2], alpha=alpha, keep_prob = keep_prob)
        x_hat = tf.layers.dense(fc3, data.shape[1] - 2, activation=None) # -2, Remove the Gender representation !

        return x_hat


# In[16]:


def accuracy_in_file(d1,d2,Train_Acc, Test_Acc, Train_d1, Train_d2, Test_d1, Test_d2, Test_Age, Test_Work, Test_Education,
                     Test_Marital, Test_occ, Test_Race, Test_Native, Test_Relationship, Diff_train, Diff_test, accuracy_sal_train, accuracy_sal_test):

    filename = "Details_Classifier_in_GAP.csv"
    file_exists = os.path.isfile(filename)

    with open(filename, 'a') as csvfile:
        fieldnames  = ["Categorical_Distortion_Upper_Bound", "Continuous_Distortion_Upper_Bound", "Train_Acc", "Test_Acc",
                  "Train_Distortion_Cat_Achieved", "Train_Distortion_Con_Achieved", "Test_Distortion_Cat_Achieved",
                  "Test_Distortion_Con_Achieved", "Test_Age_D", "Test_Work_D", "Test_Education_D", "Test_Marital_D",
                  "Test_Occ_D", "Test_Race_D", "Test_Native_D", "Test_Relationship_D", "Train_Diff", "Test_Diff", "Sal_Train", "Sal_Test"]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow({"Categorical_Distortion_Upper_Bound": str(d1), "Continuous_Distortion_Upper_Bound" : str(d2), "Train_Acc" : str(Train_Acc), "Test_Acc" : str(Test_Acc),
                  "Train_Distortion_Cat_Achieved" : str(Train_d1), "Train_Distortion_Con_Achieved" : str(Train_d2), "Test_Distortion_Cat_Achieved" : str(Test_d1),
                  "Test_Distortion_Con_Achieved" : str(Test_d2), "Test_Age_D" : str(Test_Age), "Test_Work_D" : str(Test_Work), "Test_Education_D" : str(Test_Education),
                  "Test_Marital_D" : str(Test_Marital), "Test_Occ_D" : str(Test_occ), "Test_Race_D" : str(Test_Race), "Test_Native_D" : str(Test_Native),
                  "Test_Relationship_D" : str(Test_Relationship), "Train_Diff" : str(Diff_train), "Test_Diff" : str(Diff_test), "Sal_Train" : str(accuracy_sal_train),
                         "Sal_Test" : str(accuracy_sal_test)})


# In[17]:


def test_ff_nn(Xtest, Ytest, Y_sal ,Size):
    Ytest_onehot = Ytest

    Ztest = np.random.normal(0.0, 1.0, [Size, noise_seed_dim])

    xhattest, cat_d, con_d, age_d, work_d, edu_d, mar_d, occ_d, race_d, native_d, relationship_d = sess.run([X_hat_desire, categorical_distortion_overall,
    continuous_distortion, age_log, work_log, education_log, marital_log, occupation_log, race_log, native_country_log, relationship_log],
    feed_dict={X: Xtest, Y_onehot: Ytest_onehot, sal_onehot : Y_sal, Z: Ztest, p_keep_prob: 1.0, a_keep_prob: 1.0})

    # converted_one_hot_vector = convert_real_numbers_to_one_hot(xhattest)

    ytest = sess.run(y, feed_dict={X_hat: xhattest, Y_onehot: Ytest_onehot, sal_onehot: Y_sal, Z: Ztest, p_keep_prob: 1.0, a_keep_prob: 1.0})

    ydec = np.argmax(ytest, axis=1)
    ytrue = np.argmax(Ytest, axis = 1)
    err_rate = np.mean(ytrue != ydec)

    accuracy_gender  = 1 - err_rate

    ysaltest = sess.run(sal, feed_dict={X_hat: xhattest, Y_onehot: Ytest_onehot, sal_onehot: Y_sal, Z: Ztest,
                                   p_keep_prob: 1.0, a_keep_prob: 1.0})

    ydec = np.argmax(ysaltest, axis=1)
    ytrue = np.argmax(Y_sal, axis=1)
    err = np.mean(ytrue != ydec)

    accuracy_sal = 1 - err

    male_x, female_x = male_female_data_seperate(xhattest, Ytest, Y_sal)

    male_sal = sess.run(sal, feed_dict={X_hat: male_x, sal_onehot: Y_sal, p_keep_prob: 1.0, a_keep_prob: 1.0})
    female_sal = sess.run(sal, feed_dict={X_hat: female_x, sal_onehot: Y_sal, p_keep_prob: 1.0, a_keep_prob: 1.0})

    p_m = 0
    p_f = 0
    for i in range(male_sal.shape[0]):
        if (male_sal[i,0] > male_sal[i,1]):
            p_m = p_m + 1
    for i in range(female_sal.shape[0]):
        if (female_sal[i, 0] > female_sal[i, 1]):
            p_f = p_f + 1

    diff = abs(float(p_m/male_sal.shape[0]) - float(p_f/female_sal.shape[0]))

    # print ("Male PR " + str(float(p_m/male_sal.shape[0])))
    # print ("Female PR " + str(float(p_f / female_sal.shape[0])))

    return diff, accuracy_gender, accuracy_sal, xhattest, ydec , cat_d, con_d, age_d, work_d, edu_d, mar_d, occ_d, race_d, native_d, relationship_d


# In[18]:


def convert_real_numbers_to_one_hot(input_vector):
    #This a hard coding function, where I am converting the real values generated by the privatizer to one-hot
    #encoding and then feeding it to the adversary!
    #print (input_vector[0:2,0:9+7])

    output_one_hot_vector = np.zeros((input_vector.shape[0], input_vector.shape[1]))
    counter = 0
    
    #Age one_hot 
    age_index = np.argmax(input_vector[:,0:9], axis = 1)
    counter = counter + 9

    #work_class
    work_index = np.argmax(input_vector[:,counter:counter+9], axis = 1)
    work_index = np.add(work_index, counter)
    counter = counter + 9
    
    #education
    education_index = np.argmax(input_vector[:,counter:counter+16], axis = 1)
    education_index = np.add(education_index, counter)
    counter = counter + 16
    
    #marital status 
    marital_index = np.argmax(input_vector[:,counter:counter+7], axis = 1)
    marital_index = np.add(marital_index, counter)
    counter = counter + 7

    #occupation 
    occupation_index = np.argmax(input_vector[:,counter:counter+15], axis = 1)
    occupation_index = np.add(occupation_index, counter)    
    counter = counter + 15

    #race 
    race_index = np.argmax(input_vector[:,counter:counter+5], axis = 1)
    race_index = np.add(race_index, counter)    
    counter = counter + 5

    #native_country 
    native_country_index = np.argmax(input_vector[:,counter:counter+42], axis = 1)
    native_country_index = np.add(native_country_index, counter)    
    counter = counter + 42
            
    #relationship
    relationship_index = np.argmax(input_vector[:,counter:counter+6], axis = 1)
    relationship_index = np.add(relationship_index, counter)    
    counter = counter + 6

    for i in range(input_vector.shape[0]):
        output_one_hot_vector[i,age_index[i]] = 1 
        output_one_hot_vector[i,work_index[i]] = 1 
        output_one_hot_vector[i,education_index[i]] = 1 
        output_one_hot_vector[i,marital_index[i]] = 1 
        output_one_hot_vector[i,occupation_index[i]] = 1 
        output_one_hot_vector[i,race_index[i]] = 1 
        output_one_hot_vector[i,native_country_index[i]] = 1 
        output_one_hot_vector[i,relationship_index[i]] = 1

    output_one_hot_vector[:, counter: counter+4] = np.maximum(input_vector[:, counter: counter+4],0)

    return output_one_hot_vector


# In[19]:


def euclidian_distortion(X_hat_desire, X):
    counter = 0

    # Age softmax
    age_log = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(X[:, 0:9], X_hat_desire[:, 0:9])), 1))
    counter = counter + 9

    # work_class
    work_log = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(X[:, counter: counter + 7], X_hat_desire[:, counter: counter + 7])), 1))
    counter = counter + 7

    # education
    education_log = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(X[:, counter: counter + 16], X_hat_desire[:, counter: counter + 16])), 1))
    counter = counter + 16

    # marital status
    marital_log = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(X[:, counter: counter + 7], X_hat_desire[:, counter: counter + 7])), 1))
    counter = counter + 7

    # occupation
    occupation_log = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(X[:, counter: counter + 14], X_hat_desire[:, counter: counter + 14])), 1))
    counter = counter + 14

    # race
    race_log = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(X[:, counter: counter + 5], X_hat_desire[:, counter: counter + 5])), 1))
    counter = counter + 5

    # native_country
    native_country_log = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(X[:, counter: counter + 41], X_hat_desire[:, counter: counter + 41])), 1))
    counter = counter + 41

    # relationship
    relationship_log = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(X[:, counter: counter + 6], X_hat_desire[:, counter: counter + 6])), 1))
    counter = counter + 6

    continuous_distortion = tf.reduce_mean(tf.square(tf.subtract(X[:, counter: counter + 4], X_hat_desire[:, counter: counter + 4])))

    return age_log, work_log, education_log, marital_log, occupation_log, race_log, native_country_log, relationship_log, continuous_distortion


# In[20]:


def log_loss_for_distortion(X_hat, X):
    counter = 0

    # Age softmax
    age_log = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=X[:,0:9], logits= X_hat[:,0:9]))
    counter = counter + 9

    # work_class
    work_log = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=X[:, counter:counter + 9], logits= X_hat[:, counter:counter + 9]))
    counter = counter + 9

    # education
    education_log = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=X[:, counter:counter + 16], logits= X_hat[:, counter:counter + 16]))
    counter = counter + 16

    # marital status
    marital_log = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=X[:, counter:counter + 7], logits= X_hat[:, counter:counter + 7]))
    counter = counter + 7

    # occupation
    occupation_log = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=X[:, counter:counter + 15], logits= X_hat[:, counter:counter + 15]))
    counter = counter + 15

    # race
    race_log = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=X[:, counter:counter + 5], logits= X_hat[:, counter:counter + 5]))
    counter = counter + 5

    # native_country
    native_country_log =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=X[:, counter:counter + 42], logits= X_hat[:, counter:counter + 42]))
    counter = counter + 42

    # relationship
    relationship_log = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=X[:, counter:counter + 6], logits= X_hat[:, counter:counter + 6]))
    counter = counter + 6

    continuous_distortion = tf.reduce_mean(
        tf.reduce_sum(tf.square(tf.subtract(X[:, counter: counter + 4], tf.nn.relu(X_hat[:, counter: counter + 4]))), 1))


    return age_log, work_log, education_log, marital_log, occupation_log, race_log, native_country_log, relationship_log, continuous_distortion


# In[21]:


def putting_individual_softmax(X_hat):
    counter = 0
    age_hat = tf.nn.softmax(X_hat[:, 0:9])
    counter = counter + 9

    work_hat = tf.nn.softmax(X_hat[:, counter:counter + 9])
    counter = counter + 9

    education_hat = tf.nn.softmax(X_hat[:, counter:counter + 16])
    counter = counter + 16

    marital_hat = tf.nn.softmax(X_hat[:, counter:counter + 7])
    counter = counter + 7

    occupation_hat = tf.nn.softmax(X_hat[:, counter:counter + 15])
    counter = counter + 15

    race_hat = tf.nn.softmax(logits=X_hat[:, counter:counter + 5])
    counter = counter + 5

    native_hat = tf.nn.softmax(logits=X_hat[:, counter:counter + 42])
    counter = counter + 42

    relationship_hat = tf.nn.softmax(X_hat[:, counter:counter + 6])
    counter = counter + 6

    x_hat_desire = tf.concat([age_hat, work_hat, education_hat, marital_hat, occupation_hat, race_hat, native_hat, relationship_hat, tf.nn.relu(X_hat[:, counter : counter + 4])], 1)

    return x_hat_desire


# In[22]:


def one_hot_vector_conversion(a):
    counter = 0
    max_inds = tf.argmax(a[:, 0:9], axis=1)
    inds = tf.range(0, 9, dtype=max_inds.dtype)[None, :]
    bmask = tf.equal(inds, max_inds[:, None])
    imask = tf.where(bmask, tf.ones_like(a[:, 0:9]), tf.zeros_like(a[:, 0:9]))
    counter = counter + 9

    max_inds = tf.argmax(a[:, counter:counter + 9], axis=1)
    inds = tf.range(0, 9, dtype=max_inds.dtype)[None, :]
    bmask = tf.equal(inds, max_inds[:, None])
    imask1 = tf.where(bmask, tf.ones_like(a[:, counter:counter + 9]), tf.zeros_like(a[:, counter:counter + 9]))
    counter = counter + 9

    max_inds = tf.argmax(a[:, counter:counter + 16], axis=1)
    inds = tf.range(0, 16, dtype=max_inds.dtype)[None, :]
    bmask = tf.equal(inds, max_inds[:, None])
    imask2 = tf.where(bmask, tf.ones_like(a[:, counter:counter + 16]), tf.zeros_like(a[:, counter:counter + 16]))
    counter = counter + 16

    max_inds = tf.argmax(a[:, counter:counter + 7], axis=1)
    inds = tf.range(0, 7, dtype=max_inds.dtype)[None, :]
    bmask = tf.equal(inds, max_inds[:, None])
    imask3 = tf.where(bmask, tf.ones_like(a[:, counter:counter + 7]), tf.zeros_like(a[:, counter:counter + 7]))
    counter = counter + 7

    max_inds = tf.argmax(a[:, counter:counter + 15], axis=1)
    inds = tf.range(0, 15, dtype=max_inds.dtype)[None, :]
    bmask = tf.equal(inds, max_inds[:, None])
    imask4 = tf.where(bmask, tf.ones_like(a[:, counter:counter + 15]), tf.zeros_like(a[:, counter:counter + 15]))
    counter = counter + 15

    max_inds = tf.argmax(a[:, counter:counter + 5], axis=1)
    inds = tf.range(0, 5, dtype=max_inds.dtype)[None, :]
    bmask = tf.equal(inds, max_inds[:, None])
    imask5 = tf.where(bmask, tf.ones_like(a[:, counter:counter + 5]), tf.zeros_like(a[:, counter:counter + 5]))
    counter = counter + 5

    max_inds = tf.argmax(a[:, counter:counter + 42], axis=1)
    inds = tf.range(0, 42, dtype=max_inds.dtype)[None, :]
    bmask = tf.equal(inds, max_inds[:, None])
    imask6 = tf.where(bmask, tf.ones_like(a[:, counter:counter + 42]), tf.zeros_like(a[:, counter:counter + 42]))
    counter = counter + 42

    max_inds = tf.argmax(a[:, counter:counter + 6], axis=1)
    inds = tf.range(0, 6, dtype=max_inds.dtype)[None, :]
    bmask = tf.equal(inds, max_inds[:, None])
    imask7 = tf.where(bmask, tf.ones_like(a[:, counter:counter + 6]), tf.zeros_like(a[:, counter:counter + 6]))
    counter = counter + 6

    # max_inds = tf.argmax(a[:, counter + 4:counter + 6 + 4], axis=1)
    # inds = tf.range(0, 2, dtype=max_inds.dtype)[None, :]
    # bmask = tf.equal(inds, max_inds[:, None])
    # imask8 = tf.where(bmask, tf.ones_like(a[:, counter + 4:counter + 6 + 4]),
    #                   tf.zeros_like(a[:, counter + 4:counter + 6 + 4]))

    input = tf.concat(
        values=[imask, imask1, imask2, imask3, imask4, imask5, imask6, imask7, a[:, counter:counter + 4]],
        axis=1)

    return input


# In[23]:


X_hat = privatizernn(X, Z, keep_prob = p_keep_prob)

X_hat_desire = putting_individual_softmax(X_hat)

# one_hot = one_hot_vector_conversion(X_hat_desire)
one_hot = X_hat_desire

y = adversarynn(one_hot, N_label, keep_prob = a_keep_prob)

sal = sal_classifier(one_hot, N_label, keep_prob = a_keep_prob)

sal_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=sal_onehot, logits=sal))

adversary_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_onehot, logits=y))

age_log, work_log, education_log, marital_log, occupation_log, race_log, native_country_log, relationship_log, continuous_distortion = log_loss_for_distortion(X_hat, X)  #tf.reduce_mean(tf.reduce_sum(tf.square(X-X_hat), axis=[0]))

overall_distortion = age_log +  work_log  +  education_log + marital_log + occupation_log + race_log + native_country_log + relationship_log + continuous_distortion

categorical_distortion_overall =  age_log +  work_log  +  education_log + marital_log + occupation_log + race_log + native_country_log + relationship_log

categorical_margin = tf.square(tf.maximum(categorical_distortion_overall - d1, 0))
continuous_distortion_margin = tf.square(tf.maximum(continuous_distortion - d2, 0))

privatizer_dist = (penalty_rate* (categorical_margin + continuous_distortion_margin))
privatizer_loss = privatizer_dist + (-1*adversary_loss) +  lambda_ * sal_loss


# In[24]:


t_vars = tf.trainable_variables()
a_vars = [var for var in t_vars if var.name.startswith("adversary")]
p_vars = [var for var in t_vars if var.name.startswith("privatizer")]
s_vars = [var for var in t_vars if var.name.startswith("sal_clssifier")]

a_train = tf.train.AdamOptimizer(learning_ratea).minimize(adversary_loss, var_list= a_vars)
p_train = tf.train.AdamOptimizer(learning_ratep).minimize(privatizer_loss, var_list = p_vars)
s_train = tf.train.AdamOptimizer(learning_ratep).minimize(sal_loss, var_list = s_vars)


# In[25]:


iterations  = []
saver = tf.train.Saver()


# In[26]:


# epoch = 10


# In[27]:


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    prate_base = countinous_bound
    flag = False

    for e in range(epoch):
        log_append_list = []
        X_data_shuffle, Y_data_shuffle, sal_shuffle = shuffle_data(X_data_train, Y_data_train, Y_data_sal_train)

        iterator = 0
        pe_loss = 0
        ae_loss = 0
        de = 0

        ag_d = 0
        wo_d = 0
        ed_d = 0
        me_d = 0
        occ_d = 0
        ra_d = 0
        na_d = 0
        re_d = 0
        c_d = 0
        sal_loss_log = 0
        start_time = time.time()
        for iter_ in range(num_iter):
#             print("ITER : : ", iter_)
            if iter_==(num_iter-1):
                batchX = X_data_shuffle[iterator:, :]
                batchY1 = Y_data_shuffle[iterator:, :]
                batch_sal = sal_shuffle[iterator:, :]
                batchZ = np.random.normal(0, 1, [batchX.shape[0], noise_seed_dim])
            else:
                batchX = X_data_shuffle[iterator:iterator+mbsize , :]
                batchY1 = Y_data_shuffle[iterator:iterator+mbsize , :]
                batch_sal = sal_shuffle[iterator:iterator+mbsize , :]
                batchZ = np.random.normal(0, 1, [mbsize, noise_seed_dim])


            _, A_loss_curr = sess.run([a_train, adversary_loss],
                                      feed_dict={X: batchX, Y_onehot: batchY1,  sal_onehot: batch_sal , Z: batchZ, a_keep_prob: 0.5, p_keep_prob: 1.0})


            _, P_loss_curr, p_distortion, age_log1, work_log1, education_log1, marital_log1, occupation_log1, race_log1, native_country_log1, relationship_log1, continuous_distortion1 = sess.run([p_train, privatizer_loss, overall_distortion, age_log, work_log, education_log, marital_log, occupation_log, race_log, native_country_log, relationship_log, continuous_distortion],
                                                    feed_dict={X: batchX, Y_onehot: batchY1, Z: batchZ, sal_onehot : batch_sal,
                                                               a_keep_prob: 0.5, p_keep_prob: 1.0, penalty_rate: prate_base}) #prate})

            _, s_loss_curr = sess.run([s_train, sal_loss],
                                      feed_dict={X: batchX, Y_onehot: batchY1, sal_onehot: batch_sal, Z: batchZ,
                                                 a_keep_prob: 0.5, p_keep_prob: 1.0})


            pe_loss = P_loss_curr + pe_loss
            ae_loss = A_loss_curr + ae_loss #Adversary loss
            sal_loss_log = sal_loss_log + s_loss_curr
            de = de + p_distortion
            ag_d = ag_d + age_log1
            wo_d = wo_d + work_log1
            ed_d = ed_d  + education_log1
            me_d = me_d + marital_log1
            occ_d = occ_d + occupation_log1
            ra_d = ra_d + race_log1
            na_d = na_d + native_country_log1
            re_d = re_d + relationship_log1
            c_d = c_d + continuous_distortion1
            iterator = iterator + mbsize

        
#         print("################################## Epoch : " + str(e))
# #         print("Time taken : " + str(duration))
#         print('Privatizer loss: ' + str(pe_loss/num_iter))
#         print ("Sal Classifier Loss : " + str(sal_loss_log/num_iter))
#         print("Adversary_Loss : " + str(ae_loss/num_iter))
#         print('Distortion:' + str(de/num_iter))
#         print("Age_Distortion : " + str(ag_d/num_iter))
#         print("Work_Distortion : " + str(wo_d/num_iter))
#         print("Education_Distortion : " + str(ed_d/num_iter))
#         print("Marital_Distortion : " + str(me_d/num_iter))
#         print("Occupation_Distortion : " + str(occ_d/num_iter))
#         print("Race_Distortion : " + str(ra_d/num_iter))
#         print("Native_country_Distortion : " + str(na_d/num_iter))
#         print("Relationship_Distortion : " + str(re_d/num_iter))
#         print("Continous_Distortion : " + str(c_d/num_iter))
#         print("##################################")
#         print("")
        if e % 30 == 0:
            prate_base = prate_base * 1.2

        p_loss.append(pe_loss/num_iter)
        p_dist.append(de/num_iter)

#         time_after_epoch = time.time()
        diff_train, acc_train, accuracy_sal_train, privatized_train, decision_train, cat_d, con_d, age_d, work_d, edu_d, mar_d, occ_d, race_d, native_d, relationship_d = test_ff_nn(X_data_train, Y_data_train, Y_data_sal_train, N_train)
        diff_test ,acc_test, accuracy_sal_test, privatized_test, decision_test, cat_d_t, con_d_t, age_d_t, work_d_t, edu_d_t, mar_d_t, occ_d_t, race_d_t, native_d_t, relationship_d_t = test_ff_nn(X_data_test, Y_data_test, Y_data_sal_test, N_test)
        duration = time.time() - start_time
        
        log_append_list = [e, str(duration), str(pe_loss/num_iter), str(sal_loss_log/num_iter), str(ae_loss/num_iter),
                                str(de/num_iter), str(ag_d/num_iter), str(wo_d/num_iter), str(ed_d/num_iter), str(me_d/num_iter),
                                str(occ_d/num_iter), str(ra_d/num_iter), str(na_d/num_iter), str(re_d/num_iter), str(c_d/num_iter),
                                diff_train, acc_train, accuracy_sal_train, cat_d, con_d, age_d, work_d, edu_d, mar_d, occ_d, race_d, native_d, relationship_d,
                                diff_test ,acc_test, accuracy_sal_test, cat_d_t, con_d_t, age_d_t, work_d_t, edu_d_t, mar_d_t, occ_d_t, race_d_t, native_d_t, relationship_d_t
                               ]
#         for i,x in enumerate(log_append_list):
#             print(i)
#             print(x)
#             print("\n\n")
#         print(log_append_list)
        append_record(log_append_list)
        log_append_list = []
        # print ("Gender Train Accuracy : " + str(acc_train))
        # print ("Gender Test Accuracy : " + str(acc_test))

        # print ("Sal Train Accuracy : " + str(accuracy_sal_train))
        # print ("Sal Test Accuracy : " + str(accuracy_sal_test))

        # print ("Pr Difference :" + str(diff_train))
        # print ("Pr Difference :" + str(diff_test))
#         time_taken__ = time.time() - time_after_epoch
#         print("TTTT::: ", time_taken__)
#         accuracy_in_file(d1, d2, acc_train, acc_test, cat_d, con_d, cat_d_t, con_d_t, age_d_t, work_d_t,
#                          edu_d_t, mar_d_t, occ_d_t, race_d_t, native_d_t, relationship_d_t, diff_train, diff_test, accuracy_sal_train, accuracy_sal_test)


# In[ ]:




