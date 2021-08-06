'''
UCI GAP for Alpha loss version

In order to run this code,
    python UCI_GAP_2_con.py 1.2 0.2 4

d1 : 1.2 (Categorical features uppper bound) 
d2 : 0.2 (Continuous feature upper bound)
penalty : 4 (Distortion penalty constant)


@Written by Maunil
'''


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

#alpha = 0.96 # For the alpha loss

#tf.random.set_random_seed(1234) # In case one wants to setup a seed value

def read_mat_file(filename):
    '''
    Reading the .mat file
    :param filename:
    :return:
    '''
    file_content = sio.loadmat(filename)
    X_train = file_content['X_train']
    Y_train = file_content['Y_train']
    Y_train_sal = file_content['Y_train_sal']

    return X_train, Y_train, Y_train_sal

def male_female_data_seperate(X_test, Y_test, Y_salary, d1, d2, epoch):
    '''
    Storing the final representation after GAP. Specifically, seperating Male and Female which can later used to compute the demographic parity

    :param X_test:
    :param Y_test:
    :param Y_salary:
    :param d1:
    :param d2:
    :return:
    '''
    male_x = []
    male_s = []
    female_s = []
    female_x = []
    print(X_test.shape)
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

    sio.savemat("Two_Constraints_Alpha_"+str(alpha)+"/UCI_Adult_Generated_male - " + str(d1) + "_" + str(d2) + "_epoch_" + str(epoch) + ".mat", {'X_train': male_x , 'Y_train': male_s})
    sio.savemat("Two_Constraints_Alpha_"+str(alpha)+"/UCI_Adult_Generated_female - " + str(d1) + "_" + str(d2) + "_epoch_" + str(epoch) + ".mat", {'X_train': female_x , 'Y_train': female_s})

def save_the_train_data_set(X_hat_data_train, X_hat_data_test, Y_data_train, Y_data_test, d1, d2, epoch):
    '''
    Storing the GAP represenation as a .mat file

    :param X_hat_data_train:
    :param X_hat_data_test:
    :param Y_data_train:
    :param Y_data_test:
    :param d1:
    :param d2:
    :return:
    '''
    directory = "Two_Constraints_Alpha_"+str(alpha)
    if os.path.isdir(directory):
        pass
    else:
        os.mkdir(directory)

    sio.savemat("Two_Constraints_Alpha_" + str(alpha) + "/UCI_Adult_Generated - " + str(d1) + "_" + str(d2) + "_epoch_" + str(epoch) + ".mat",
                {'X_train': X_hat_data_train, 'Y_train': Y_data_train, 'X_test': X_hat_data_test,
                 'Y_test': Y_data_test})


X_data_train, Y_data_train, Y_data_sal_train = read_mat_file("UCI_Adult_Train_data_for_GAP_Complete.mat")  # X_train : 30162 x 109
X_data_test, Y_data_test, Y_data_sal_test = read_mat_file("UCI_Adult_Test_data_for_GAP_Complete.mat")  # X_train : 30162 x 109

N_train = X_data_train.shape[0]
N_test = X_data_test.shape[0]

mbsize = 500
epoch = 300
learning_ratea = 0.001
learning_ratep = 0.001

d1 = float(sys.argv[1]) # This one is for the categorical features
d2 = float(sys.argv[2]) # This one is for the continuous features
#alpha = float(sys.argv[3]) # This one is for the alpha value of the alpha loss function!
#alpha = 1.2
countinous_bound = float(sys.argv[3]) # This is to set the penalty rate!
alpha = float(sys.argv[4]) # This one is for the alpha value of the alpha loss function!

overall_distortion_value = d1 + d2

print ("#Distortion Upper Bound")
print (d1)
print (d2)
print ("Alpha :" + str(alpha))
noise_seed_dim = 113
num_iter = int(N_train/mbsize)

print("Total interations:", num_iter)

#N_act = 10
N_label = Y_data_train.shape[1]
N_feature = X_data_train.shape[1]
N_all = N_train + N_test

a_loss = []
p_loss = []
p_dist = []

X = tf.placeholder(dtype = tf.float32, shape = [None, N_feature])
Y_onehot = tf.placeholder(tf.float32, shape=[None, N_label])
Z = tf.placeholder(dtype = tf.float32, shape=[None, noise_seed_dim], name='Z')
keep_prob = tf.placeholder(tf.float32)
penalty_rate = tf.placeholder(tf.float32)
p_keep_prob = tf.placeholder(tf.float32)
a_keep_prob = tf.placeholder(tf.float32)

def shuffle_data(X_data_train, Y_data_train):
    '''
    shuffling the data for each mini batch

    :param X_data_train:
    :param Y_data_train:
    :return:
    '''
    idx = np.arange(X_data_train.shape[0])
    np.random.shuffle(idx)
    x_data = [X_data_train[i,:] for i in idx]
    y_data = [Y_data_train[i,:] for i in idx]
    return np.asarray(x_data) , np.asarray(y_data)

def adversarynn(data, num_out, structure = [10, 5], alpha = learning_ratea, keep_prob = 1.0):
    '''
    Adversary model

    :param data:
    :param num_out:
    :param structure:
    :param alpha:
    :param keep_prob:
    :return:
    '''
    with tf.variable_scope("adversary"):
        fc1_a = fc_bn_leakyRelu(data, structure[0], alpha = alpha, keep_prob = keep_prob)
        fc2_a = fc_bn_leakyRelu(fc1_a, structure[1], alpha=alpha, keep_prob = keep_prob)
        h_hat = tf.layers.dense(fc2_a, num_out, activation=None)
        return h_hat

def privatizernn(data, noise_seed, structure=[170, 130, 113], alpha=learning_ratep, keep_prob = 1.0):
    '''
    Privatizer model of the GAP

    :param data:
    :param noise_seed:
    :param structure:
    :param alpha:
    :param keep_prob:
    :return:
    '''

    with tf.variable_scope("privatizer"):
        input = tf.concat(values=[data, noise_seed], axis=  1)
        fc1 = fc_bn_leakyRelu(input, structure[0], alpha=alpha, keep_prob = keep_prob)
        fc2 = fc_bn_leakyRelu(fc1, structure[1], alpha=alpha, keep_prob = keep_prob)
        fc3 = fc_bn_leakyRelu(fc2, structure[2], alpha=alpha, keep_prob = keep_prob)
        x_hat = tf.layers.dense(fc3, data.shape[1], activation=None)

        return x_hat


def accuracy_in_file(d1,d2,Train_Acc, Test_Acc, Train_d1, Train_d2, Test_d1, Test_d2, Test_Age, Test_Work, Test_Education, Test_Marital, Test_occ, Test_Race, Test_Native, Test_Relationship, e):
    '''
    To track the accuracy and other parameters after every epoch - Mainly to monitor the performance

    :param d1:
    :param d2:
    :param Train_Acc:
    :param Test_Acc:
    :param Train_d1:
    :param Train_d2:
    :param Test_d1:
    :param Test_d2:
    :param Test_Age:
    :param Test_Work:
    :param Test_Education:
    :param Test_Marital:
    :param Test_occ:
    :param Test_Race:
    :param Test_Native:
    :param Test_Relationship:
    :return:
    '''

    directory = "Two_Constraints_Alpha_" + str(alpha)
    if os.path.isdir(directory):
        pass
    else:
        os.mkdir(directory)

    filename = "Two_Constraints_Alpha_"+str(alpha)+"/Gender_Accuracy_Details.csv"
    file_exists = os.path.isfile(filename)

    with open(filename, 'a') as csvfile:
        fieldnames  = ["Categorical_Distortion_Upper_Bound", "Continuous_Distortion_Upper_Bound", "Train_Acc", "Test_Acc",
                  "Train_Distortion_Cat_Achieved", "Train_Distortion_Con_Achieved", "Test_Distortion_Cat_Achieved",
                  "Test_Distortion_Con_Achieved", "Test_Age_D", "Test_Work_D", "Test_Education_D", "Test_Marital_D",
                  "Test_Occ_D", "Test_Race_D", "Test_Native_D", "Test_Relationship_D", "Epoch"]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow({"Categorical_Distortion_Upper_Bound": str(d1), "Continuous_Distortion_Upper_Bound" : str(d2), "Train_Acc" : str(Train_Acc), "Test_Acc" : str(Test_Acc),
                  "Train_Distortion_Cat_Achieved" : str(Train_d1), "Train_Distortion_Con_Achieved" : str(Train_d2), "Test_Distortion_Cat_Achieved" : str(Test_d1),
                  "Test_Distortion_Con_Achieved" : str(Test_d2), "Test_Age_D" : str(Test_Age), "Test_Work_D" : str(Test_Work), "Test_Education_D" : str(Test_Education),
                  "Test_Marital_D" : str(Test_Marital), "Test_Occ_D" : str(Test_occ), "Test_Race_D" : str(Test_Race), "Test_Native_D" : str(Test_Native),
                  "Test_Relationship_D" : str(Test_Relationship), "Epoch" : str(e)})


def test_ff_nn(Xtest, Ytest, Size):
    '''
    Computing the accuracy - Gender mainly

    :param Xtest:
    :param Ytest:
    :param Size:
    :return:
    '''

    Ytest_onehot = Ytest

    Ztest = np.random.normal(0.0, 1.0, [Size, noise_seed_dim])

    xhattest, cat_d, con_d, age_d, work_d, edu_d, mar_d, occ_d, race_d, native_d, relationship_d = sess.run([X_hat, categorical_distortion_overall,
    continuous_distortion, age_log, work_log, education_log, marital_log, occupation_log, race_log, native_country_log, relationship_log],
    feed_dict={X: Xtest, Y_onehot: Ytest_onehot, Z: Ztest, p_keep_prob: 1.0, a_keep_prob: 1.0})

    converted_one_hot_vector = convert_real_numbers_to_one_hot(xhattest)

    ytest = sess.run(y, feed_dict={X_hat: converted_one_hot_vector, Y_onehot: Ytest_onehot, Z: Ztest, p_keep_prob: 1.0, a_keep_prob: 1.0})

    ydec = np.argmax(ytest, axis=1)
    ytrue = np.argmax(Ytest, axis = 1)
    err_rate = np.mean(ytrue != ydec)

    accuracy = 1 - err_rate
    return accuracy, converted_one_hot_vector, ydec , cat_d, con_d, age_d, work_d, edu_d, mar_d, occ_d, race_d, native_d, relationship_d

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

def euclidian_distortion(X_hat_desire, X):
    '''
    Distortion computation using euclidean

    :param X_hat_desire:
    :param X:
    :return:
    '''
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


def log_loss_for_distortion(X_hat, X):
    '''
    Distortion computation with log loss

    :param X_hat:
    :param X:
    :return:
    '''
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

    #cross_entropy_distortion = age_log + work_log + education_log + marital_log + occupation_log + race_log + native_country_log + relationship_log + continuous_distortion

    return age_log, work_log, education_log, marital_log, occupation_log, race_log, native_country_log, relationship_log, continuous_distortion


def putting_individual_softmax(X_hat):
    '''
    Adding inividual softmax to each category features to get values in probabilities


    :param X_hat:
    :return:
    '''
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

def alpha_loss(label, logits, alpha):
    '''
    Alpha loss function

    :param label:
    :param logits:
    :param alpha:
    :return:
    '''
    epsilon = 1e-05 # This is to make sure that the softmax logits not become zero
    logits = logits + epsilon
    soft_max_logits = tf.nn.softmax(logits)
    loss_sum = (alpha / (alpha - 1)) * tf.reduce_mean((label *(1 - tf.pow(soft_max_logits, 1 - (1 / alpha)))))
    return (loss_sum*2)

def graph_plot(a_loss, p_loss, d, epoch, d1, d2, alpha):
    e = [int(i) for i in range(epoch)]

    plt.figure (1)
    plt.plot(e , a_loss, 'bo--', label='Adversary Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig("Adversary Loss_"  +str(d1) + "_" + str(d2) + "_alpha_" + str(alpha) + ".png")

    plt.figure(2)
    plt.plot(e , p_loss, 'g+--', label='Privatizer Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig("Privatizer Loss_" +str(d1) + "_" + str(d2) + "_alpha_" + str(alpha) + ".png")

    plt.figure(3)
    plt.plot(e , d, 'g+--', label='Distortion')
    plt.xlabel("Epoch")
    plt.ylabel("Distortion")
    plt.legend()
    plt.grid()
    plt.savefig("Distortion_"  +str(d1) + "_" + str(d2) + "_alpha_" + str(alpha) + ".png")



'''
Notice that Tensorflow first builds a graph and then later we feed the data to the graph. The below mentioned lines creates 
the Neural Network graph. Later, session.run () onwards we feed the data to this graph and train the GAP 
'''

X_hat = privatizernn(X, Z, keep_prob = p_keep_prob) #Mapping the placeholders to privatizer

X_hat_desire = putting_individual_softmax(X_hat)    #Mapping the output of the privatizer to softmax

y = adversarynn(X_hat_desire, N_label, keep_prob = a_keep_prob) # Mapping the adversary with the output of the privatizer

#adversary_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_onehot, logits=y)) # To use conventional cross - entropy loss for the gender

adversary_loss = alpha_loss(Y_onehot, y, alpha) # Alpha loss for the gender (Adversary)

# Checking the distortion of the inidividual components, log loss for the categorical features and euclidean for the continuous features
age_log, work_log, education_log, marital_log, occupation_log, race_log, native_country_log, relationship_log, continuous_distortion = log_loss_for_distortion(X_hat, X)  #tf.reduce_mean(tf.reduce_sum(tf.square(X-X_hat), axis=[0]))

# Summing up the overall distortion
overall_distortion = age_log +  work_log  +  education_log + marital_log + occupation_log + race_log + native_country_log + relationship_log + continuous_distortion

# Summing up the categorical distortion to see whether it is below the distortion bound
categorical_distortion_overall =  age_log +  work_log  +  education_log + marital_log + occupation_log + race_log + native_country_log + relationship_log

# Making un constraint problem using lagrange multiplier ideas
categorical_margin = tf.square(tf.maximum(categorical_distortion_overall - d1, 0))
continuous_distortion_margin = tf.square(tf.maximum(continuous_distortion - d2, 0))

# Privatizer loss computation, two components -  (1) Distortion minimize (2) Adversary loss maximize (negative sign added)
privatizer_dist = (penalty_rate* (categorical_margin + continuous_distortion_margin))
privatizer_loss = privatizer_dist + (-1*adversary_loss)

# Creating trainble variables and assigning to adversary and privatizer
t_vars = tf.trainable_variables()
a_vars = [var for var in t_vars if var.name.startswith("adversary")]
p_vars = [var for var in t_vars if var.name.startswith("privatizer")]

# Setting up Adam optimizer for privatizer and adversary
a_train = tf.train.AdamOptimizer(learning_ratea).minimize(adversary_loss, var_list= a_vars)
p_train = tf.train.AdamOptimizer(learning_ratep).minimize(privatizer_loss, var_list = p_vars)

iterations  = []
saver = tf.train.Saver()

# Training starts - feeding data to the Tensor graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    prate_base = countinous_bound
    flag = False
    adversary_run = 5

    # Running for epochs
    for e in range(epoch):
        X_data_shuffle, Y_data_shuffle = shuffle_data(X_data_train, Y_data_train) # Mini - batch get
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

        for iter in range(num_iter):
            start_time = time.time()
            if iter==(num_iter-1):
                batchX = X_data_shuffle[iterator:, :]
                batchY1 = Y_data_shuffle[iterator:, :]
                batchZ = np.random.normal(0, 1, [batchX.shape[0], noise_seed_dim])
            else:
                batchX = X_data_shuffle[iterator:iterator+mbsize , :]
                batchY1 = Y_data_shuffle[iterator:iterator+mbsize , :]
                batchZ = np.random.normal(0, 1, [mbsize, noise_seed_dim])
            
            A_loss_curr = 0
            for j in range(adversary_run):
                _, dummy_A_loss_curr = sess.run([a_train, adversary_loss],
                                      feed_dict={X: batchX, Y_onehot: batchY1,  Z: batchZ, a_keep_prob: 0.5, p_keep_prob: 1.0})
                A_loss_curr = A_loss_curr + dummy_A_loss_curr
            
            A_loss_curr = float(A_loss_curr/adversary_run)
            
            # Privatizer training mode
            _, P_loss_curr, p_distortion, age_log1, work_log1, education_log1, marital_log1, occupation_log1, race_log1, native_country_log1, relationship_log1, continuous_distortion1 = sess.run([p_train, privatizer_loss, overall_distortion, age_log, work_log, education_log, marital_log, occupation_log, race_log, native_country_log, relationship_log, continuous_distortion],
                                                    feed_dict={X: batchX, Y_onehot: batchY1, Z: batchZ,
                                                               a_keep_prob: 0.5, p_keep_prob: 1.0, penalty_rate: prate_base}) #prate})


            pe_loss = P_loss_curr + pe_loss
            ae_loss = A_loss_curr + ae_loss #Adversary loss
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

        print("################################## Epoch : " + str(e))
        print('Privatizer loss: ' + str(pe_loss/num_iter))
        print('Distortion:' + str(de/num_iter))
        print("Age_Distortion : " + str(ag_d/num_iter))
        print("Work_Distortion : " + str(wo_d/num_iter))
        print("Education_Distortion : " + str(ed_d/num_iter))
        print("Marital_Distortion : " + str(me_d/num_iter))
        print("Occupation_Distortion : " + str(occ_d/num_iter))
        print("Race_Distortion : " + str(ra_d/num_iter))
        print("Native_country_Distortion : " + str(na_d/num_iter))
        print("Relationship_Distortion : " + str(re_d/num_iter))
        print("Continous_Distortion : " + str(c_d/num_iter))
        duration = time.time() - start_time
        print('adversary_loss: iter=%d, loss=%f, time=%.3f' % (iter, (ae_loss/num_iter), duration))
        print("##################################")
        print("")

        # if e > 200:
        #     prate_base = prate_base + 2

        if e % 30 == 0:
            prate_base = prate_base * 1.2

        p_loss.append(pe_loss/num_iter)
        p_dist.append(de/num_iter)
        a_loss.append(ae_loss/num_iter)
        
        # Computing the accuracies
        acc_train, privatized_train, decision_train, cat_d, con_d, age_d, work_d, edu_d, mar_d, occ_d, race_d, native_d, relationship_d = test_ff_nn(X_data_train, Y_data_train, N_train)
        acc_test, privatized_test, decision_test, cat_d_t, con_d_t, age_d_t, work_d_t, edu_d_t, mar_d_t, occ_d_t, race_d_t, native_d_t, relationship_d_t = test_ff_nn(X_data_test, Y_data_test, N_test)
        
        print ("Train Accuracy (Gender): " + str(acc_train))
        print ("Test Accuracy (Gender): " + str(acc_test))
        print ("Train Cat Distortion: " + str(cat_d))
        print ("Train Con Distortion: " + str(con_d))
        print ("Test Cat Distortion: " + str(cat_d_t))
        print ("Test Con Distortion: " + str(con_d_t))
        
        if (cat_d <= d1 and con_d <= d2 and acc_test >= 0.65):
            accuracy_in_file(d1, d2, acc_train, acc_test, cat_d, con_d, cat_d_t, con_d_t, age_d_t, work_d_t,
                         edu_d_t, mar_d_t, occ_d_t, race_d_t, native_d_t, relationship_d_t, e)
            
            save_the_train_data_set(privatized_train, privatized_test, Y_data_sal_train, Y_data_sal_test, d1, d2, e)
            male_female_data_seperate(privatized_test, Y_data_test, Y_data_sal_test, d1, d2, e)

    # Computing the accuracies
    #acc_train, privatized_train, decision_train, cat_d, con_d, age_d, work_d, edu_d, mar_d, occ_d, race_d, native_d, relationship_d = test_ff_nn(X_data_train, Y_data_train, N_train)
    #acc_test, privatized_test, decision_test, cat_d_t, con_d_t, age_d_t, work_d_t, edu_d_t, mar_d_t, occ_d_t, race_d_t, native_d_t, relationship_d_t = test_ff_nn(X_data_test, Y_data_test, N_test)
    
    graph_plot(a_loss, p_loss, p_dist, epoch, d1, d2, alpha) # Change this function - Need to somehow store the python list results and plot in your local machine 

    #print ("Train Accuracy (Gender): " + str(acc_train))
    #print ("Test Accuracy (Gender): " + str(acc_test))

    # Storing accuracy in a file
    #accuracy_in_file(d1, d2, acc_train, acc_test, cat_d, con_d, cat_d_t, con_d_t, age_d_t, work_d_t,
    #                 edu_d_t, mar_d_t, occ_d_t, race_d_t, native_d_t, relationship_d_t)


    # Saving the newly generated representation!!
    #save_the_train_data_set(privatized_train, privatized_test, Y_data_sal_train, Y_data_sal_test, d1, d2)
    #male_female_data_seperate(privatized_test, Y_data_test, Y_data_sal_test, d1, d2)

    #Done!!