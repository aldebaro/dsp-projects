'''Trains a deep NN for whistled speech classification.

Assumes the input X and output y arrays are in a single HDF5 file.

See, for explanation about convnet and filters:
https://datascience.stackexchange.com/questions/16463/what-is-are-the-default-filters-used-by-keras-convolution2d
and
http://cs231n.github.io/convolutional-networks/
'''
from __future__ import print_function

from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, MaxPool2D, Reshape
#from keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Concatenate, Bidirectional, LSTM
import numpy as np
import tensorflow as tf
from tensorflow import keras
import sys
import h5py
import os
import json
import pandas as pd
import argparse

from ml_util import *

#one can disable the imports below if not plotting / saving
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

######## Main parameters ########
epochs = 4500 #number of epochs (passing over training set)
batch_size = 3 #mini-batch size when calculating the gradient
validationFraction = 0.1 #fraction from data used for validation

def train_and_test(X_train, y_train, X_test, y_test, num_rows, num_columns, output_file_name, continue_training):
    #add extra dimension to the end, as suggested by Keras
    input_shape = (num_rows, num_columns, 1)
    X_train = tf.expand_dims(X_train, -1)
    X_test = tf.expand_dims(X_test, -1)

    #Convert output labels to one-hot encoding according to
    #https://stackoverflow.com/questions/65328084/one-hot-encode-labels-of-tf-data-dataset
    #numClasses = 8
    #label = tf.one_hot(label, numClasses, name='label', axis=-1)

    if continue_training:
        #load model (architecture + weights)
        model = tf.keras.models.load_model(output_file_name)
    else:
        #create model
        if use_daniel_model:
            model = define_model_Daniel(input_shape)
        else:
            model = define_model(input_shape)
    model.summary()

    #not using one-hot encoding for labels, so use sparse_categorical_crossentropy
    model.compile(loss=keras.losses.sparse_categorical_crossentropy,
                #loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])
                
    #install graphviz: sudo apt-get install graphviz and then pip install related packages
    #plot_model(model, to_file='classification_model.png', show_shapes = True)

    #about early stopping: https://keras.io/api/callbacks/early_stopping/
    callbacks = [
        #tf.keras.callbacks.EarlyStopping(monitor='loss', patience=100, verbose=1, restore_best_weights=True)
        #tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, verbose=1, restore_best_weights=True)
        #keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"), #save models as it goes
    ]

    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        #callbacks=callbacks, 
                        verbose=1,
                        shuffle=True,
                        validation_split=validationFraction)
                        #validation_data=(X_test, y_test))

    # print results
    score = model.evaluate(X_test, y_test, verbose=1)
    #print(model.metrics_names)
    #print('Test loss rmse:', np.sqrt(score[0]))
    #print('Test accuracy:', score[1])
    #print(score)

    #print("Contents in history:", history.history.keys())

    #save accuracies for plotting in external programs, could also save the losses:
    if False:
        val_acc = history.history['val_accuracy']
        acc = history.history['accuracy']
        f = open('classification_output.txt','w')
        f.write('validation_acc\n')
        f.write(str(val_acc))
        f.write('\ntrain_acc\n')
        f.write(str(acc))
        f.close()

    #save NN:
    #https://www.tensorflow.org/guide/keras/save_and_serialize
    model.save(output_file_name)
    print("Wrote file ", output_file_name)

    #enable if want to plot images
    if show_images:
        from keras.utils import plot_model
        import matplotlib.pyplot as plt

        pred_test = model.predict(X_test)
        for i in range(len(y_test)):
            if (y_test[i] != np.argmax(pred_test[i])):
                myImage = X_test[i] #.reshape(nrows,ncolumns)
                plt.imshow(myImage)
                plt.show()
                print(np.unique(X_test[i]))
                #print("Type <ENTER> for next")
                #input()

    return score

if __name__ == '__main__':
    print("=====================================")
    print("Train DNN")

    parser = argparse.ArgumentParser()
    parser.add_argument('--general_dir',help='folder with wavs_labels.csv and labels_dictionary.json files',default='../general')
    parser.add_argument('--output_file',help='output file for saving the neural network model')
    parser.add_argument('--show_plot', action='store_true') #default is false
    parser.add_argument('--output_dir',help='output folder (default is the folder of the input file)')
    parser.add_argument('--disjoint_speakers', action='store_true') #default is false
    parser.add_argument('--daniel_model', action='store_true') #default is false
    parser.add_argument('--continue_training', action='store_true') #default is false
    #required arguments    
    parser.add_argument('--input_file',help='input file with features',required=True)
    args = parser.parse_args()

    show_images = args.show_plot #show images
    instances_file = args.input_file
    general_dir = args.general_dir
    label_file = os.path.join(general_dir, "wavs_labels.csv")
    labels_dic_file = os.path.join(general_dir, "labels_dictionary.json")
    use_disjoint_speakers = args.disjoint_speakers
    continue_training = args.continue_training
    use_daniel_model = args.daniel_model #use the CNN arquitecture proposed by Daniel et al

    if continue_training and use_disjoint_speakers:
        print("ERROR: continue training has not been implemented for disjoint speakers")
        exit(1)

    if args.output_dir == None:
        #create a default name
        output_folder = os.path.dirname(instances_file)
    else:
        output_folder = args.output_dir

    if not os.path.exists(output_folder):
        os.makedirs(output_folder,exist_ok=True) #create folder if it does not exist
        print("Created output folder",output_folder)

    #name of output file with model
    if args.output_file == None:
        #create a default name
        output_dnn_file_name = os.path.join(output_folder,
        "nn_model_Daniel" + str(use_daniel_model) + "Disjoint" + str(use_disjoint_speakers) + "_" + os.path.basename(instances_file))
    else:
        output_dnn_file_name = args.output_file
        folder_to_write = os.path.dirname(output_dnn_file_name)
        if folder_to_write == "":
            #user provided just the file name
            output_dnn_file_name = os.path.join(output_folder, output_dnn_file_name)
        else:
            if not os.path.exists(folder_to_write):
                print("Folder", folder_to_write, "does not exist. Correct argument --output_file")
                exit(-1)

    #print(output_dnn_file_name)

    if (not continue_training) and os.path.exists(output_dnn_file_name):
        print("ERROR: ", output_dnn_file_name, "already exists!")
        print("Delete this file or choose another name with command line parameter --output_file")
        exit(-1)

    #read the labels_file
    df_input_data = pd.read_csv(label_file) #read CSV label file using Pandas
    num_examples = len(df_input_data)
    if num_examples != 127:
        print("WARNING!!!! num_examples=", num_examples)

    #save this script in the output folder for reproducibility
    if os.name == 'nt':
        command = "copy " + sys.argv[0] + " " + output_folder
    else: #Linux
        command = "cp " + sys.argv[0] + " " + output_folder
    os.system(command)
    print("Executed: ", command)

    X, y = read_instances_from_file(instances_file)
    y = y.astype(int) #convert output labels to integers
    #num_examples, D, T = X.shape
    num_examples, T, D = X.shape #we use the transpose
    #print(num_examples, D, T)
    num_rows = T
    num_columns = D

    a_file = open(labels_dic_file, "r")
    label_dict_str = a_file.read()
    #https://appdividend.com/2022/01/29/how-to-convert-python-string-to-dictionary/
    label_dict = json.loads(label_dict_str)
    print("labels_dict", label_dict)
    print("labels_dict", type(label_dict))

    all_scores = list()
    #split into train and test correctly
    if use_disjoint_speakers:
        speaker_indices_dic, all_speakers = split_into_train_test_disjoint_speakers(df_input_data)
        num_speakers = len(all_speakers)
        for i in range(num_speakers):
            test_speaker = all_speakers[i]
            print("############## Training model for test speaker", test_speaker)
            X_train, X_test, y_train, y_test, train_indices, test_indices = compose_train_test_disjoint_speakers(X, y, speaker_indices_dic, all_speakers, test_speaker)

            this_output_dnn_file_name = os.path.splitext(output_dnn_file_name)[0] + "_no_" + test_speaker + ".hdf5"
            score = train_and_test(X_train, y_train, X_test, y_test, num_rows, num_columns, this_output_dnn_file_name, continue_training)
            all_scores.append(score)
        print(all_speakers)
        print('Test scores')
        print(all_scores)
    else:
        #allow mixed speakers in train and test
        if continue_training:
            file_with_test_indices = os.path.join(output_folder,"test_indices.csv")
            test_indices = np.genfromtxt(file_with_test_indices, delimiter=',')
            test_indices = test_indices.astype(int)
            file_with_train_indices = os.path.join(output_folder,"train_indices.csv")
            train_indices = np.genfromtxt(file_with_train_indices, delimiter=',')                
            train_indices = train_indices.astype(int)
            print("train_indices", train_indices, type(train_indices))
            X_test = X[test_indices]
            y_test = y[test_indices]
            X_train = X[train_indices]
            y_train = y[train_indices]
        else:
            test_fraction = 0.2
            X_train, X_test, y_train, y_test, train_indices, test_indices = split_into_train_test_mixed_speakers(X, y, test_fraction)
        score = train_and_test(X_train, y_train, X_test, y_test, num_rows, num_columns, output_dnn_file_name, continue_training)
        file_with_test_indices = os.path.join(output_folder,"test_indices.csv")
        np.savetxt(file_with_test_indices, test_indices, delimiter=",")
        file_with_train_indices = os.path.join(output_folder,"train_indices.csv")
        np.savetxt(file_with_train_indices, train_indices, delimiter=",")
