import keras
from keras import applications, optimizers
from keras.metrics import sparse_categorical_accuracy
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Flatten, Dropout
from keras.callbacks import ModelCheckpoint
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

path1 = '/home/mjpatel8/9-9/'
class Prep_UTK:
    def __init__(self, path):
        self.path = path
        self.gender_classes = 2
        self.race_classes = 4

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
        #self.load_from_json(path)

    def preprocess(self, encode=False):
        # self.load_from_json()
        self.load_from_mat()
        # if encode:
        #     self.encode_labels()
        # else:
        #     self.get_labels()
        return self.train_data, self.valid_data, self.test_data, [self.train_labels_1, self.train_labels_2, self.train_labels_3],\
                   [self.valid_labels_1, self.valid_labels_2, self.valid_labels_3], [self.test_labels_1, self.test_labels_2, self.test_labels_3]

    def encode_labels(self):

        # one hot encode train labels
        self.train_labels_1 = self.train_labels[:, 0]
        self.train_labels_2 = keras.utils.to_categorical(self.train_labels[:, 1], self.gender_classes)
        self.train_labels_3 = keras.utils.to_categorical(self.train_labels[:, 2], self.race_classes)

        # one hot encode valid labels
        self.valid_labels_1 = self.valid_labels[:, 0]
        self.valid_labels_2 = keras.utils.to_categorical(self.valid_labels[:, 1], self.gender_classes)
        self.valid_labels_3 = keras.utils.to_categorical(self.valid_labels[:, 2], self.race_classes)

        # one hot encode test labels
        self.test_labels_1 = self.test_labels[:, 0]
        self.test_labels_2 = keras.utils.to_categorical(self.test_labels[:, 1], self.gender_classes)
        self.test_labels_3 = keras.utils.to_categorical(self.test_labels[:, 2], self.race_classes)

    def load_from_json(self):
        try:
            # print('Converting Images')
            # self.train_data, self.train_labels = self.img_to_arr(128)
            # print('Done Image Resizing')

            # data = scipy.io.loadmat(self.path + 'Part1.mat')
            # data = pickle.load(open(self.path + 'train.pickle', 'rb'))

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

    def load_from_mat(self):
        try:
            data = scipy.io.loadmat('Data_Noisy.mat')

            self.train_data = np.array(data['privatized_train'])
            self.test_data = np.array(data['privatized_test'])
            self.valid_data = np.array(data['privatized_valid'])

            self.train_labels_1 = data['y_train_age']
            self.train_labels_1 = self.train_labels_1.reshape((self.train_labels_1.shape[1], ))
            self.train_labels_2 = data['y_train_gender']
            self.train_labels_2 = keras.utils.to_categorical(self.train_labels_2.reshape((self.train_labels_2.shape[1], )), self.gender_classes)
            self.train_labels_3 = data['y_train_race']
            self.train_labels_3 = keras.utils.to_categorical(self.train_labels_3.reshape((self.train_labels_3.shape[1], )), self.race_classes)


            #self.test_data_recon = np.array(data['x_test_recon'])

            self.test_labels_1 = data['y_test_age']
            self.test_labels_1 = self.test_labels_1.reshape((self.test_labels_1.shape[1], ))
            self.test_labels_2 = data['y_test_gender']
            self.test_labels_2 = keras.utils.to_categorical(self.test_labels_2.reshape((self.test_labels_2.shape[1], )), self.gender_classes)
            self.test_labels_3 = data['y_test_race']
            self.test_labels_3 = keras.utils.to_categorical(self.test_labels_3.reshape((self.test_labels_3.shape[1], )), self.race_classes)

            self.valid_labels_1 = data['y_valid_age']
            self.valid_labels_1 = self.valid_labels_1.reshape((self.valid_labels_1.shape[1],))
            self.valid_labels_2 = data['y_valid_gender']
            self.valid_labels_2 = keras.utils.to_categorical(self.valid_labels_2.reshape((self.valid_labels_2.shape[1],)), self.gender_classes)
            self.valid_labels_3 = data['y_valid_race']
            self.valid_labels_3 = keras.utils.to_categorical(self.valid_labels_3.reshape((self.valid_labels_3.shape[1],)), self.race_classes)

        except IOError:
            print('Incorrect File path')

    def load_from_mat2(self):
        try:
            data = scipy.io.loadmat('Output_AE.mat')

            self.train_data = np.array(data['x_train_recon'])
            self.test_data = np.array(data['x_test_recon'])
            self.valid_data = np.array(data['x_valid_recon'])

            self.train_labels_1 = data['y_train_age']
            self.train_labels_1 = self.train_labels_1.reshape((self.train_labels_1.shape[1], ))
            self.train_labels_2 = data['y_train_gender']
            self.train_labels_2 = keras.utils.to_categorical(self.train_labels_2.reshape((self.train_labels_2.shape[1], )), self.gender_classes)
            self.train_labels_3 = data['y_train_race']
            self.train_labels_3 = keras.utils.to_categorical(self.train_labels_3.reshape((self.train_labels_3.shape[1], )), self.race_classes)


            #self.test_data_recon = np.array(data['x_test_recon'])

            self.test_labels_1 = data['y_test_age']
            self.test_labels_1 = self.test_labels_1.reshape((self.test_labels_1.shape[1], ))
            self.test_labels_2 = data['y_test_gender']
            self.test_labels_2 = keras.utils.to_categorical(self.test_labels_2.reshape((self.test_labels_2.shape[1], )), self.gender_classes)
            self.test_labels_3 = data['y_test_race']
            self.test_labels_3 = keras.utils.to_categorical(self.test_labels_3.reshape((self.test_labels_3.shape[1], )), self.race_classes)

            self.valid_labels_1 = data['y_valid_age']
            self.valid_labels_1 = self.valid_labels_1.reshape((self.valid_labels_1.shape[1],))
            self.valid_labels_2 = data['y_valid_gender']
            self.valid_labels_2 = keras.utils.to_categorical(self.valid_labels_2.reshape((self.valid_labels_2.shape[1],)), self.gender_classes)
            self.valid_labels_3 = data['y_valid_race']
            self.valid_labels_3 = keras.utils.to_categorical(self.valid_labels_3.reshape((self.valid_labels_3.shape[1],)), self.race_classes)

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

class Classifier:
    def __init__(self):
        self.model = None
        self.history = None
        self.img_rows, self.img_cols, self.img_channel = 64, 64, 3
        self.create_model()

    def create_model(self):
        base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(self.img_rows, self.img_cols, self.img_channel))
        add_model = Sequential()
        add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
        add_model.add(Dense(256, activation='relu'))
        add_model.add(Dropout(0.5))
        add_model.add(Dense(4, activation='softmax'))
        self.model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                      metrics=['accuracy'])

    def train(self, xtrain, ytrain, xvalid, yvalid, batch_size = 128, epochs = 20, callbacks = []):
        self.history = self.model.fit(xtrain, ytrain, verbose = 1, batch_size = batch_size, epochs = epochs, validation_data = (xvalid, yvalid), callbacks = callbacks)

    def evaluate(self, xtest, ytest):
        return self.model.evaluate(xtest, ytest)

    def plot(self):
        # Plot training & validation accuracy values
        plt.plot(self.history.history['acc'])
        plt.plot(self.history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig('Models/Race_Model_Accuracy.png')
        plt.clf()

        # Plot training & validation loss values
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig('Models/Race_Model_Loss.png')
        plt.clf()

prep = Prep_UTK(path1)
print('-----------> Loading Data')
x_train, x_valid, x_test, y_train, y_valid, y_test = prep.preprocess(encode=True)
print('Train', x_train.shape)
print('Test', x_test.shape)
print('Valid', x_valid.shape)
print('-----------> Done Loading Data')

res = open('Race_Test_Scores.txt', 'w')
epochs = 100
classifier = Classifier()
acc_save = ModelCheckpoint('Models/Race_Best_Val_Acc_Model_' + str(epochs) + '.hdf5', save_best_only=True,
                           monitor='val_acc', mode='max')
loss_save = ModelCheckpoint('Models/Race_Best_Val_Loss_Model_' + str(epochs) + '.hdf5', save_best_only=True,
                           monitor='val_loss', mode='min')
classifier.train(x_train, y_train[2], x_valid, y_valid[2],epochs=epochs, callbacks = [acc_save, loss_save])
classifier.model.save('Models/Race_Model_' + str(epochs) + '.hdf5')
classifier.plot()
classifier = load_model('Models/Race_Best_Val_Acc_Model_' + str(epochs) + '.hdf5')
score = classifier.evaluate(x_test, y_test[2])
res.write('Race_For best valid acc')
print('Race_Test loss is : %r' % (score[0]))
res.write('Race_Test loss is : %r \n' % (score[0]))
print('Race_Test accuracy is : %r' % (score[1]))
res.write('Race_Test accuracy is : %r \n' % (score[1]))

classifier = load_model('Models/Race_Best_Val_Loss_Model_' + str(epochs) + '.hdf5')
score = classifier.evaluate(x_test, y_test[2])
res.write('Race_For best valid loss')
print('Race_Test loss is : %r' % (score[0]))
res.write('Race_Test loss is : %r \n' % (score[0]))
print('Race_Test accuracy is : %r' % (score[1]))
res.write('Race_Test accuracy is : %r \n' % (score[1]))

res.close()
