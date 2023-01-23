'''Machine learning useful code.
'''
from __future__ import print_function

from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, MaxPool2D, Reshape
#from keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Concatenate, Bidirectional, LSTM
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
import sys
import h5py
import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

#one can disable the imports below if not plotting / saving
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import pickle

dropout_probability = 0.0 #to combat overfitting
numClasses = 8 #total number of labels

'''
Get indices for each speaker to later build speaker-disjoint sets
'''
def split_into_train_test_disjoint_speakers(df_input_data):
    all_speakers = df_input_data.speaker.unique()
    print(all_speakers)
    num_speakers = len(all_speakers)
    speaker_indices_dic = {}
    #initialize lists
    for i in range(num_speakers):
        speaker_indices_dic[all_speakers[i]] = list()
    #populate lists
    for i, this_row in df_input_data.iterrows():
        wavfile = this_row['filename']
        speaker = this_row['speaker']
        speaker_indices_dic[speaker].append(i)
    #print(speaker_indices_dic)
    return speaker_indices_dic, all_speakers

'''
Separate indices to create disjoint speaker sets.
'''
def compose_train_test_disjoint_speakers(X, y, speaker_indices_dic, all_speakers, test_speaker):
    num_speakers = len(all_speakers)
    train_indices = list()
    for i in range(num_speakers):
        speaker = all_speakers[i]
        if speaker == test_speaker:
            test_indices = speaker_indices_dic[test_speaker]
        else:
            train_indices.extend(speaker_indices_dic[speaker])
    print('train_indices=',train_indices)
    print('test_indices=',test_indices)
    trainX = X[train_indices]
    testX = X[test_indices]
    trainy = y[train_indices]
    testy = y[test_indices]
    return trainX, testX, trainy, testy, train_indices, test_indices

'''
Do not pay attention to the speaker and let them get mixed into training and test sets
X can be a list or a numpy array.
'''
def split_into_train_test_mixed_speakers(X, y, test_fraction):
    if test_fraction < 0 or test_fraction > 1:
        raise Exception("test_fraction=", test_fraction)    
    if type(X) == list: #X can be a list or a numpy array
        num_examples = len(X)
    else:
        num_examples = X.shape[0]
    n_train = int(np.round((1.0 - test_fraction) * num_examples))
    indices = np.arange(num_examples)
    #https://stackoverflow.com/questions/17649875/why-does-random-shuffle-return-none
    #new_indices = np.random.shuffle(indices)
    new_indices = random.sample(list(indices), len(indices))
    train_indices = new_indices[:n_train]
    test_indices = new_indices[n_train:]
    if type(X) == list: #X can be a list or a numpy array
        #https://stackoverflow.com/questions/18272160/access-multiple-elements-of-list-knowing-their-index
        trainX = [ X[i] for i in train_indices]
        testX = [ X[i] for i in test_indices]
    else:
        trainX = X[train_indices]
        testX = X[test_indices]
    trainy = y[train_indices]
    testy = y[test_indices]
    #print(train_indices, test_indices)
    return trainX, testX, trainy, testy, train_indices, test_indices

'''
Do not pay attention to the speaker and let them get mixed into training and test sets.
And do not keep track of the indices with respect to the input.
'''
def split_into_train_test_mixed_speakers_unknow_indices(X, y, test_fraction):
    if test_fraction < 0 or test_fraction > 1:
        raise Exception("test_fraction=", test_fraction)
    trainX, testX, trainy, testy = train_test_split(X, y, test_size=test_fraction, stratify=True)
    return trainX, testX, trainy, testy

'''
Read X and y from HDF5 file.
'''
def read_instances_from_file(intputHDF5FileName):
    h5pyFile = h5py.File(intputHDF5FileName, 'r')
    Xtemp = h5pyFile["X"]
    ytemp = h5pyFile["y"]
    X = np.array(Xtemp[()])
    y = np.array(ytemp[()])
    h5pyFile.close()
    return X, y

'''
Write X and y to HDF5 file.
'''
def write_instances_to_file(outputHDF5FileName, X, y):
    h5pyFile = h5py.File(outputHDF5FileName, 'w')
    h5pyFile["X"] = X #store in hdf5
    h5pyFile["y"] = y #store in hdf5
    h5pyFile.close()
    #print('==> Wrote file ', outputHDF5FileName, ' with keys X and y')

'''
Read X from pickle files and create y.
X is a list with variable dimension arrays.
The input dimension is D x T, but the output is T x D (transposed).
'''
def read_variable_duration_instances_from_files(input_folder, df_input_data, label_dict_str):
    n = len(df_input_data)
    print("Will process", n, "files")
    X = list()
    y = np.zeros( (n,), dtype=int)
    for i, row in df_input_data.iterrows():        
        wavfile_basename = row['filename']
        this_label = row['annotation']
        num_samples = row['stops']
        file_name = os.path.splitext(wavfile_basename)[0] + "_fea.pkl"
        file_name = os.path.join(input_folder, file_name)
        with open(file_name, 'rb') as f:
            this_X = pickle.load(f)
        print(wavfile_basename, this_label, "num_samples=", num_samples, "final shape=", this_X.shape)
        #print(type(label_dict_str), label_dict_str,"ajad", this_label)
        y[i] = label_dict_str[this_label]
        X.append(this_X.T) #use transpose when dealing with the machine learning
    return X, y


'''
Define CNN.
'''
def define_model(input_shape):

    num_filter_base = 10

    model = Sequential()
    model.add(Conv2D(num_filter_base, kernel_size=(10,10),
                    activation='tanh',
                    #strides=[1,1],
                    padding="SAME",
                    #consider we have matrices images with depth = 1:
                    input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))                 
    model.add(Dropout(dropout_probability))
    model.add(Conv2D(num_filter_base, (6, 6), padding="SAME", activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))                 
    #model.add(Dropout(dropout_probability))
    #model.add(Conv2D(4*num_filter_base, (4, 4), padding="SAME", activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))                 
    model.add(Conv2D(num_filter_base, (3, 3), padding="SAME", activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))                 
    model.add(Dropout(dropout_probability))
    model.add(Flatten())
    model.add(Dense(numClasses, activation='softmax')) #softmax for probability

    return model

'''
Encoder part of the autoencoder from Daniel paper, without the LSTM.
'''
def define_model_Daniel_no_LSTM(in_shape):
    conv_params   = [
        (8, 8,  32),
        (4, 16, 32),
        (2, 32, 32),
        (1, 64, 32),
        (8,  4, 32),
        (16, 4, 32),
        (32, 4, 32)
    ]
    dft_dim = in_shape[1]
    shape = (None, dft_dim, 1)
    inp = Input(shape)

    convolutions = []
    for w, h, n_filters in conv_params:
        kernel_size = (w, h)
        loc = Conv2D(n_filters, strides = (1, 1), kernel_size=kernel_size, activation='relu', padding='same')(inp) 
        loc = MaxPool2D(pool_size=(1, dft_dim))(loc)
        loc = Reshape((-1, n_filters))(loc) 
        convolutions.append(loc)  
    loc = Concatenate(axis=2)(convolutions)
    #print(loc)
    #https://stackoverflow.com/questions/53439765/in-keras-how-to-use-reshape-layer-with-none-dimension
    loc = Reshape((1, 67200))(loc) #TO-DO make this number automatically chosen
    z = Flatten()(loc)
    #x = Dense(20*numClasses, activation='softmax')(z)
    #y = Flatten()(x)
    z = Dense(numClasses, activation='softmax')(z)
    #x   = Bidirectional(LSTM(latent_dim, return_sequences=True))(loc)
    return Model(inputs =[inp], outputs=[z])

'''
Encoder part of the autoencoder from Daniel paper (with the LSTM).
'''
def define_model_Daniel(in_shape):
    latent_dim = 128
    conv_params   = [
        (8, 8,  32),
        (4, 16, 32),
        (2, 32, 32),
        (1, 64, 32),
        (8,  4, 32),
        (16, 4, 32),
        (32, 4, 32)
    ]
    dft_dim = in_shape[1]
    shape = (None, dft_dim, 1)
    inp = Input(shape)

    convolutions = []
    for w, h, n_filters in conv_params:
        kernel_size = (w, h)
        loc = Conv2D(n_filters, strides = (1, 1), kernel_size=kernel_size, activation='relu', padding='same')(inp) 
        loc = MaxPool2D(pool_size=(1, dft_dim))(loc)
        loc = Reshape((-1, n_filters))(loc) 
        convolutions.append(loc)  
    loc = Concatenate(axis=2)(convolutions)
    x  = Bidirectional(LSTM(latent_dim, return_sequences=False))(loc)
    x = Flatten()(x)
    z = Dense(numClasses, activation='softmax')(x)
    return Model(inputs =[inp], outputs=[z])

