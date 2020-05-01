###############################################################################
#
# AUTHOR(S): Samantha Muellner
#            Josh Kruse
# DESCRIPTION: program that will implement a stochastic gradient descent algo
#       for a neural network with one hidden layer
# VERSION: 1.3.0v
#
###############################################################################

import numpy as np
import csv, math
from math import sqrt
from matplotlib import pyplot as plt
import random

from sklearn.preprocessing import scale
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from keras.layers.core import Flatten, Dropout, Lambda
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras import optimizers
from keras.utils import to_categorical
import tensorflow as tf


#from SG_with_early_stopping_regularization import SG_main
#from nearest_neightbors import NN_main
#from gradientDescent import GD_main

# global variables
MAX_EPOCHS = 20
DATA_FILE = "zip.train"
VAL_SPLIT = 0.2


# Function: split matrix
# INPUT ARGS:
#   X_mat : matrix to be split
#   y_vec : corresponding vector to X_mat
# Return: train, validation, test
def split_matrix(X_mat, y_vec, size):
    # split data 80% train by 20% validation
    X_train, X_validation = np.split( X_mat, [int(size * len(X_mat))])
    y_train, y_validation = np.split( y_vec, [int(size * len(y_vec))])

    return (X_train, X_validation, y_train, y_validation)


# Function: convert_data_to_matrix
# INPUT ARGS:
#   file_name : the csv file that we will be pulling our matrix data from
# Return: data_matrix_full
def convert_data_to_matrix(file_name):
    #with open(file_name, 'r') as data_file:
    #    spam_file = list(csv.reader(data_file, delimiter = " "))

    #data_matrix_full = np.array(spam_file[0:], dtype=np.float)

    # read data from csv
    all_data = np.genfromtxt(DATA_FILE, delimiter=" ")

    # set inputs to everything but first col, and scale

    X = np.asarray(np.delete(all_data, 0, axis=1))

    # set outputs to first col of data
    y = np.asarray(all_data[:, 0])

    return X, y

# Function: sigmoid
# INPUT ARGS:
#   x : value to be sigmoidified
# Return: sigmoidified x
def sigmoid(x) :
    x = 1 / (1 + np.exp(-x))
    return x


# function that will create our NN model given the amount of units passed in
def create_fully_model() :
    sgd = optimizers.SGD(lr=0.01, clipnorm=1.)
    
    model = Sequential()
    
    model.add(Dense(units=784, activation='sigmoid', use_bias=False))
    model.add(Dense(units=270, activation='sigmoid', use_bias=False))
    model.add(Dense(units=270, activation='sigmoid', use_bias=False))
    model.add(Dense(units=128, activation='sigmoid', use_bias=False))
    model.add(Dense(10, activation="sigmoid", use_bias=False))
    
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    return model

# function thay will create our CNN model
def create_convo_model() :
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(16, 16, 1), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))

    model.compile(loss='categorical_crossentropy', optimizer='Adadelta',
                  metrics=['accuracy'])
    return model


def generate_color( seed ):
    random.seed( seed )
    color = '#{:02x}{:02x}{:02x}'.format(*map(lambda x: random.randint(0, 255), range(3)))
    return color

# function to plot our loss
def plot_loss(res, vec, title):
    best = [0, 0, 0]

    for index, item in enumerate(res):
        plt.plot(item.history['loss'], label=str(vec[index]) + " train", color=generate_color(index))
        min_index = np.argmin(item.history['loss'])
        plt.plot(min_index, item.history['loss'][min_index], "go")

        #plt.plot(item.history['val_loss'], '--', label=str(vec[index]) + " val", color=generate_color(index))
        #res_best = np.argmin(item.history['val_loss'])
        #res_loss = np.min(item.history['val_loss'])
        #plt.plot(res_best, item.history['val_loss'][res_best], "go")

        #print(res_loss)

        #if min_index > best[2]:
        #    best[0] = index
        #    best[1] = res_best
        #    best[2] = res_loss

    plt.title('model loss with respect to ' + title)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper left')
    plt.show()

    return best


# Function: main
def main():
    print("starting")
    # use spam data set

    X_sc, y_vec = convert_data_to_matrix(DATA_FILE)

    #np.random.seed( 0 )
    #np.random.shuffle(data_matrix_full)

    # get necessary variables
    # shape yields tuple : (row, col)
    #col_length = data_matrix_full.shape[1]

    #X_Mat = np.delete(data_matrix_full, col_length - 1, 1)
    #y_vec = data_matrix_full[:,57]

    #X_sc = scale(X_Mat)

    # (10 points) For 5-fold cross-validation, create a variable fold_vec 
    #   which randomly assigns each observation to a fold from 1 to 5.

    num_folds = 5
    multiplier_of_num_folds = int(X_sc.shape[0]/num_folds)

    is_train = np.array(list(np.arange(1,
                                        num_folds + 1))
                                        * multiplier_of_num_folds)

    
    # make sure that test_fold_vec is the same size as X_Mat.shape[0]
    while is_train.shape[0] != X_sc.shape[0]:
        is_train = np.append(is_train, random.randint(1, num_folds))


    fully_matrix_list = []
    convo_matrix_list = []

    # (10 points) For each fold ID, you should create variables x_train, 
    #   y_train, x_test, y_test based on fold_vec.
    for test_fold in range(1, num_folds + 1):

        X_new = np.delete( X_sc, np.argwhere( is_train != test_fold ), 0 )
        y_new = np.delete( y_vec, np.argwhere( is_train != test_fold ), 0 )

        X_test = np.delete(X_sc, np.argwhere(is_train == test_fold), 0)
        X_test_convo = X_test.reshape(X_test.shape[0], 16, 16, 1)
        y_test = np.delete(y_vec, np.argwhere(is_train == test_fold), 0)
        y_test = to_categorical(y_test)

        subtrain_size = np.sum( is_train == test_fold )
        is_subtrain = np.random.choice( [True, False], subtrain_size, p=[.8, .2] )

        X_train = np.delete( X_new, np.argwhere( is_subtrain != True ), 0)
        X_train_convo = X_train.reshape(X_train.shape[0], 16, 16, 1)

        y_train = np.delete( y_new, np.argwhere( is_subtrain != True ), 0)
        y_train = to_categorical(y_train)

        X_validation = np.delete( X_new, np.argwhere( is_subtrain != False ), 0)
        X_validation_convo = X_validation.reshape(X_validation.shape[0], 16, 16, 1)

        y_validation = np.delete( y_new, np.argwhere( is_subtrain != False ), 0)
        y_validation = to_categorical(y_validation)

        print(X_train.shape)
        print(X_validation.shape)
        print(is_subtrain.shape)

        #X_test = np.delete( X_sc, np.argwhere( is_train != False ), 0 )
        #y_test = np.delete( y_vec, np.argwhere( is_train != False ), 0 )

        fully_model = create_fully_model()
        # train on x-train, y-train
        # save results to data table (split_matrix_list) for further analysis
        fully_history = fully_model.fit( x = X_train,
                                y = y_train,
                                epochs = MAX_EPOCHS,
                                validation_data=(X_validation, y_validation),
                                verbose=2)

        convo_model = create_convo_model()
        convo_history = convo_model.fit( x = X_train_convo,
                                                  y = y_train,
                                                  epochs = MAX_EPOCHS,
                                                  validation_data = (X_validation_convo, y_validation),
                                                  verbose = 2)

        best_fully_epoch = np.argmin(fully_history.history['val_loss'])
        best_convo_epoch = np.argmin(convo_history.history['val_loss'])

        fully_final_model = create_fully_model()
        # train on x-train, y-train
        # save results to data table (split_matrix_list) for further analysis
        fully_final_model.fit(x=X_train,
                                                 y=y_train,
                                                 epochs=best_fully_epoch,
                                                 verbose=2)
        fully_matrix_list.append(fully_final_model.evaluate(X_test, y_test)[1])

        convo_final_model = create_convo_model()

        convo_final_model.fit(x=X_train_convo,
                                                 y=y_train,
                                                 epochs=best_convo_epoch,
                                                 verbose=2)
        convo_matrix_list.append(convo_final_model.evaluate( X_test_convo, y_test )[1])

    # (10 points) Also compute the accuracy of the baseline model, which always
    #   predicts the most frequent class label in the train data.

    print(fully_matrix_list)
    print(convo_matrix_list)
    # (10 points) At the end of your for loop over fold IDs, you should store 
    #   the accuracy values, model names, and fold IDs in a data structure (e.g. 
    #   list of data tables) for analysis/plotting.

    # (10 points) Finally, make a dotplot that shows all 15 test accuracy 
    #   values. The Y axis should show the different models, and the X axis 
    #   should show the test accuracy values. There should be three rows/y axis 
    #   ticks (one for each model), and each model have five dots (one for each 
    #   test fold ID). Make a comment in your report on your interpretation of 
    #   the figure. Are the neural networks better than baseline? Which of the 
    #   two neural networks is more accurate?

main()


# %%
