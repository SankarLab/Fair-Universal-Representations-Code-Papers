#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.image as img
# import pandas as pd
import numpy as np
from keras import applications, optimizers
from keras.metrics import sparse_categorical_accuracy
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, TerminateOnNaN
import os
from sklearn.model_selection import train_test_split
import scipy.io


# In[2]:


a = scipy.io.loadmat("Data_Noisy.mat")
epochs_ = 50


# In[3]:


X_train, X_test,X_valid, y_valid, y_train, y_test = a["privatized_train"], a["privatized_test"],a["privatized_valid"],\
                                                    a["y_valid_age"].T, a["y_train_age"].T, a["y_test_age"].T
# X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[4]:

# keras.backend.clear_session()
model = Sequential()
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', input_shape=(64,64,3)))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='linear'))
# model = K.Sequential([
#     K.layers.Conv2D(128, (3, 3), activation='relu', padding='same', input_shape=(64,64,3)),
#     K.layers.MaxPooling2D(pool_size=(3, 3)),
#     K.layers.Conv2D(64, (2, 2), activation='relu', padding='same'),
#     K.layers.MaxPooling2D(pool_size=(2, 2)),
#     K.layers.Conv2D(32, (2, 2), activation='relu', padding='same'),
#     K.layers.MaxPooling2D(pool_size=(2, 2)),
#     K.layers.Flatten(),
#     K.layers.Dense(512, activation='relu'),
#     K.layers.Dense(128, activation='relu'),
#     K.layers.Dense(1, activation='linear')
# ])


# In[5]:


model.compile(optimizer= optimizers.Adam(lr=0.001),
                                        loss='mse',
                                        metrics=['mae'])

filepath="Models/weights_logloss.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_mean_absolute_error', verbose=2, save_best_only=True, mode='min')
## the default value of 'verbose' 0: the way of printing intermedia results
callbacks_list = [checkpoint, TerminateOnNaN()]

history = model.fit(X_train,
                      y_train,
                      batch_size=128,
                      epochs=epochs_,
                      validation_data=(X_valid, y_valid),
                      callbacks=callbacks_list,
                      verbose = 1)
res = {}
file = open('Age_scores.txt','w')
model = load_model(filepath)
y_test_1d = [i[0] for i in y_test]
# df = pd.DataFrame(columns=['True', 'Predicted', 'Gender'])
res['True'] = y_test_1d
res['Gender'] = a['y_test_gender'][0]

res['Predicted'] = model.predict(X_test)
score = model.evaluate(X_test, y_test)
print(score)
file.write('Age test loss is %f and mae is %f' %(score[0], score[1]))
scipy.io.savemat('Predictions', res)
file.close()
