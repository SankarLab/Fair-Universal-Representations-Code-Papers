'''
Input : This script will take the representations from the "Two_constraints_Alpha_xx" folder.
Functions : It will train a NN classifier to learn the salary (uses alpha loss)
Outputs : It will generate "numerical(number)".csv file, which will have performance regarding Accuracy and DP for every 10 epoch. (Recommend to use this as validation is not used in the codebase)
        : It will also produces a "Salary.csv" which will have the sal details at 250 epoch
'''

import numpy as np
import scipy.io as sio 
import torch
import torch.nn.functional as F
import glob
import csv
import os
import sys

#To run on CPU uncomment the CPU device part and comment the GPU
device  = torch.device("cpu")
#device = torch.device("cuda:0")
dtype =  torch.float
dt = np.dtype(float) 

alpha = float(sys.argv[1])
alpha_for_privatizer = float(sys.argv[2])

def write_in_mat_file(Acc_test, Acc_train, Epoch, loss):
    sio.savemat("Acc_Details.mat", {'Acc_test' : Acc_test, 'Acc_train' : Acc_train , 'Epoch' : Epoch, 'loss':loss})

def read_mat_file(filename):
    file_content = sio.loadmat(filename)
    X_train = file_content['X_train']
    Y_train = file_content['Y_train']
    X_test = file_content['X_test']
    Y_test = file_content['Y_test']

    return X_train, Y_train, X_test, Y_test

def init_weights(m):
    '''
    Weight Initialization 
    One of the crucial role in tunning the network parameters. Here I used xavier's intialization for weights  
    '''
    if type(m) == torch.nn.Linear:
       torch.nn.init.xavier_normal_(m.weight, gain=1)
       #torch.nn.init.xavier_uniform(m.weight)
       #m.bias.data.fill_(0.01)

def accuracy_calculate(nn_model, Data, labels):
    '''
    This function is responsible to calculate the accracy. 
    Moreover, it also calculate the confidence of the model, means how confident the model is for the true label 
    classes. 
    '''

    #Feed Forwarding
    pred = nn_model(Data[:,:]) 
    acc = 0
    pred = torch.nn.functional.softmax(pred, dim=1)
    #confidence = 0

    for i in range(pred.shape[0]):
        if(pred[i,0] > pred[i,1] and labels[i,0] == 0):
           acc = acc + 1
        elif(pred[i,0] <= pred[i,1] and labels[i,0] == 1.0):
           acc = acc + 1                     
      
    return ((acc*100)/pred.shape[0])

def alpha_loss(logits, label):
    '''
    Alpha loss util
    :param logits:
    :param label:
    :return:
    '''
    epsilon = 1e-05  # This is to make sure that the softmax logits not become zero
    logits = logits + epsilon
    log_prob = 1 - (F.softmax(logits, 1))**(1-(1/alpha))
    loss = log_prob.gather(1, label.unsqueeze(1))
    loss = (loss.mean())
    return loss*2

def Neural_Network_Architecture(N_shape):
    model = torch.nn.Sequential(torch.nn.Linear(N_shape[0],N_shape[1]), torch.nn.ReLU(), torch.nn.Linear(N_shape[1],N_shape[2]), torch.nn.ReLU(), torch.nn.Linear(N_shape[2],N_shape[3]))
    model.apply(init_weights)
    return model 

def NN_Training(nn_model, Train_Data, Train_label , Epoch, Batch_size, X_test, Y_test, X_male, X_female, i, upper_bound):
    #If you are running on CPU, uncomment the below line of code
    #nn_model.cuda()

    Train_Tensor = torch.from_numpy(Train_Data[:,:])
    Train_Label = torch.from_numpy(Train_label[:,:])

    #Putting tensors on Cuda device 
    Train_Tensor = torch.tensor(Train_Tensor, device = device, dtype = dtype)
    Train_Label = torch.tensor(Train_Label, device = device, dtype = dtype)

    #Loss and Optimizer 
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(nn_model.parameters(), lr=0.007)
   
    number_of_example_in_batch  = (Train_Data.shape[0]/Batch_size)

    Training_Loss = []

    epo  = []
    acc_train1 = []

    for t in range(Epoch):
        iterator = 0
        l = 0
        for k in range(int(number_of_example_in_batch)):
            if(k == number_of_example_in_batch-1):  #For the last batch, I will take all the remaining example
                train = Train_Tensor[iterator:, :]
                label = Train_Label[iterator:,0]
            else:
                train = Train_Tensor[iterator:iterator + Batch_size,:]
                label = Train_Label [iterator:iterator + Batch_size,0]

            label = torch.tensor(label, device = device, dtype = torch.long)
            y_pred = nn_model(train)
            loss = alpha_loss(y_pred, label)
            optimizer.zero_grad()
            loss.backward()
            l = l + loss.item()
            optimizer.step()
            iterator = iterator + Batch_size

        if t % 10 == 0:
            acc_train = accuracy_calculate(nn_model, Train_Tensor, Train_Label)
            acc_test = feed_forward_for_test(nn_model, X_test, Y_test)
            print ("Train Acc : " + str(acc_train) + "Test Acc : "  + str(acc_test))

            #Male_Conditional_Pr
            m_l_50k, m_g_50k = compute_the_conditional(nn_model, X_male)
            #Female_Conditional_Pr
            f_l_50k, f_g_50k = compute_the_conditional(nn_model, X_female)

            write_salary_acc(acc_train, acc_test, m_l_50k, m_g_50k, f_l_50k, f_g_50k, upper_bound, i)

        epo.append(t+1)
        print("Epoch " + str(t+1) + " Training Loss is: " + str(l/number_of_example_in_batch))
        Training_Loss.append(float(l/Batch_size))

    acc_train = accuracy_calculate(nn_model, Train_Tensor, Train_Label)
    # acc_test = accuracy_calculate(nn_model, Test_Tensor, Test_Label)
    print("Train_Accuracy : " + str(acc_train))
    # print("Test_Accuracy : "  +  str(acc_test))
    acc_train1.append(acc_train)

    return acc_train, epo, Training_Loss

def feed_forward_for_test (nn_model, X, Y):
    Test_Tensor  = torch.from_numpy(X[:,:])
    Test_Tensor = torch.tensor(Test_Tensor, device=device, dtype=dtype)
    Test_Label = torch.from_numpy(Y[:,:])
    Test_Label = torch.tensor(Test_Label, device = device, dtype = dtype)

    acc = accuracy_calculate(nn_model, Test_Tensor, Test_Label)

    return acc

def NN_Model_save(nn_model, path):
    torch.save(nn_model,path)

def get_the_data_files(): # This will give me all the file having the following name! All data files from the privatizer
    file_list = glob.glob("./Two_Constraints_Alpha_" + str(alpha_for_privatizer) + "/UCI_Adult_Generated - *.mat")
    file_list_male = glob.glob("./Two_Constraints_Alpha_" + str(alpha_for_privatizer) + "/UCI_Adult_Generated_male - *.mat")
    file_list_female = glob.glob("./Two_Constraints_Alpha_" + str(alpha_for_privatizer) + "/UCI_Adult_Generated_female - *.mat")
    return file_list, file_list_male, file_list_female

def write_salary_acc(Train_Acc, Test_Acc, m_l_50k, m_g_50k, f_l_50k, f_g_50k, distortion_bound, filename):
    filename = str(filename) + ".csv"
    file_exists = os.path.isfile(filename)
    diff = abs(float(m_l_50k) - float(f_l_50k))

    with open(filename, 'a', newline = "") as csvfile:
        fieldnames = ["distortion_bound", "Train_Acc", "Test_Acc","Diff","Male_Less", "Male_Greater", "Female_Less", "Female_Greater"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow({"distortion_bound" : str(distortion_bound), "Train_Acc" : str(Train_Acc), "Test_Acc" : str(Test_Acc), "Diff": str(diff),
                         "Male_Less" : str(m_l_50k), "Male_Greater" : str(m_g_50k), "Female_Less" : str(f_l_50k), "Female_Greater" : str(f_g_50k)})


def read(male_file, female_file):
    file_content = sio.loadmat(male_file)
    X_train_m = file_content['X_train']
    Y_train_m = file_content['Y_train']

    file_content1 = sio.loadmat(female_file)
    X_train_f = file_content1['X_train']
    Y_train_f = file_content1['Y_train']

    return X_train_m, Y_train_m, X_train_f, Y_train_f

def compute_the_conditional(nn_model , X_test):
    pred = nn_model(X_test[:, :])

    less_than_50k = 0
    for i in range(pred.shape[0]):
        if(pred[i][0] > pred[i][1]):
            less_than_50k = less_than_50k + 1

    pr_less_than_50k = (less_than_50k/pred.shape[0])

    return str(pr_less_than_50k), str(1-pr_less_than_50k)

def main():
    file_list, file_list_male, file_list_female = get_the_data_files()

    count_var = 1
    for i in range(len(file_list)):
        X_train, Y_train, X_test, Y_test = read_mat_file(file_list[i])
        X_male, S_male, X_female, S_female = read(file_list_male[i], file_list_female[i])

        X_ml = torch.from_numpy(X_male[:, :])
        X_ml = torch.tensor(X_ml, device=device, dtype=dtype)

        X_fl = torch.from_numpy(X_female[:, :])
        X_fl = torch.tensor(X_fl, device=device, dtype=dtype)

        upper_bound = (file_list[i][40:len(file_list[i])-4])

        #ANN, Neural Network architecture.
        No_Input  = X_train.shape[1]
        No_Hidden1 = 10
        No_Hidden2 = 5
        No_Output = 2  # Prediction task, Sigmoid in the last layer

        N_shape = [No_Input, No_Hidden1, No_Hidden2 , No_Output]
        nn_model = Neural_Network_Architecture(N_shape)

        #ANN, parameters
        Epoch = 200
        Batch_size = 250

        train_acc, epo, loss = NN_Training(nn_model, X_train, Y_train, Epoch, Batch_size, X_test, Y_test, X_ml, X_fl, count_var, upper_bound)
        acc_test = feed_forward_for_test(nn_model, X_test, Y_test)

        # Male_Conditional_Pr
        m_l_50k, m_g_50k = compute_the_conditional(nn_model, X_ml)
        # Female_Conditional_Pr
        f_l_50k, f_g_50k = compute_the_conditional(nn_model, X_fl)

        print ("Test Acc " + str(acc_test))

        write_salary_acc(train_acc, acc_test, m_l_50k, m_g_50k, f_l_50k, f_g_50k, upper_bound, "Salary_Details_" + str(alpha) + "_" + str(alpha_for_privatizer))
        count_var = count_var + 1

if __name__ == "__main__":
    main()