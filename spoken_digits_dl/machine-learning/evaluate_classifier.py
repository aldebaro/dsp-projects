'''Test a deep NN that was trained for whistled speech classification.

Assumes the input X and output y arrays are in a single HDF5 file
and splits train / test internally
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
import seaborn as sn
import sklearn

#one can disable the imports below if not plotting / saving
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

from ml_util import *

def save_confusion_matrix(confusion_matrix, confusion_matrix_file, class_names, test_speaker):
    #TO-DO maybe use https://github.com/wcipriano/pretty-print-confusion-matrix
    pointer = sn.heatmap(confusion_matrix, annot= True, xticklabels = class_names,
            yticklabels = class_names)
    plt.title("Test with speaker " + test_speaker)
    figure = pointer.get_figure()    

    figure.savefig(confusion_matrix_file, dpi=300)
    print("Wrote PNG file", confusion_matrix_file)
    #plt.show() #enable to show figures
    plt.clf()


def test_trained_model(X_test, y_test, test_indices, num_rows, num_columns, input_file_name, df_input_data, label_dict):

    #Convert output labels to one-hot encoding according to
    #https://stackoverflow.com/questions/65328084/one-hot-encode-labels-of-tf-data-dataset
    #numClasses = 8
    #label = tf.one_hot(label, numClasses, name='label', axis=-1)

    #load model (architecture + weights)
    model = tf.keras.models.load_model(input_file_name)

    #from https://stackoverflow.com/questions/65862711/tf-keras-how-to-get-expected-input-shape-when-loading-a-model
    config = model.get_config() # Returns pretty much every information about your model
    expected_input_shape = config["layers"][0]["config"]["batch_input_shape"]
    print("expected_input_shape", expected_input_shape) 
    #model.summary()
    #print("X_test", X_test)
    if len(expected_input_shape) > 2:
        #add extra dimension to the end, as suggested by Keras
        input_shape = (num_rows, num_columns, 1)
        X_test = tf.expand_dims(X_test, -1)
    else: #useful when we work in latent space
        input_shape = (expected_input_shape[1],)

    print("test_indices", test_indices)

    # predict all DNN outputs (dimension of each output is the number of classes):
    y_predicted = model.predict(X_test)
    #convert 8 numbers into the index of their maximum:
    y_predicted_label = np.argmax(y_predicted, axis = 1) 
    print('y_test', y_test)
    #print("AAA", y_predicted)
    print("y_label", y_predicted_label)

    #need to convert indices into labels
    #label_dict is a dictionary of labels into indices. Need to find its inverse:
    inv_label_dict = {v: k for k, v in label_dict.items()}

    #evaluate each example, whether is a correct prediction or not
    num_test_examples = X_test.shape[0]
    num_errors = 0
    for i in range(num_test_examples):
        index_referring_to_all_indices = test_indices[i]
        #print("input_index=",index_referring_to_all_indices)
        #label = df_input_data["annotation"][input_index]
        filename = df_input_data["filename"][index_referring_to_all_indices]
        #print("aa", y_predicted[i], type(y_predicted[i]))        
        predicted_label = inv_label_dict[int(y_predicted_label[i])]
        #print("y_test[i], y_label[i]", y_test[i], y_predicted_label[i])
        if y_test[i] == y_predicted_label[i]:
            print("CORRECT: ", filename, ", model predicted", predicted_label)
        else:
            print("ERROR: For ", filename, ", model predicted", predicted_label)
            num_errors += 1

    error = 100*num_errors/num_test_examples
    print("Misclassification error = ", 100*num_errors/num_test_examples, "%")

    confusion_matrix = sklearn.metrics.confusion_matrix(y_test, y_predicted_label)

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

    return error, confusion_matrix

if __name__ == '__main__':
    print("=====================================")
    print("Test DNN")

    parser = argparse.ArgumentParser()
    parser.add_argument('--general_dir',help='folder with wavs_labels.csv and labels_dictionary.json files',default='../general')
    parser.add_argument('--output_file',help='output file for saving the neural network model')
    parser.add_argument('--disjoint_speakers', action='store_true') #default is false
    parser.add_argument('--show_plot', action='store_true') #default is false
    parser.add_argument('--output_dir',help='output folder (default is the folder of the input file)')
    parser.add_argument('--test_indices_exist', help='a file called test_indices.csv in the input file folder must be used for defining the test set', action='store_true') #default is false    
    #required arguments    
    parser.add_argument('--input_file',help='input file with features',required=True)
    parser.add_argument('--input_model',help='input file with ML model (or prefix when using --disjoint_speakers)',required=True)
    args = parser.parse_args()

    test_indices_exist = args.test_indices_exist
    use_disjoint_speakers = args.disjoint_speakers
    show_images = args.show_plot #show images
    instances_file = args.input_file
    general_dir = args.general_dir
    label_file = os.path.join(general_dir, "wavs_labels.csv")
    labels_dic_file = os.path.join(general_dir, "labels_dictionary.json")
    input_dnn_file_name = args.input_model #file with DNN model

    if args.output_dir == None:
        #create a default name
        output_folder = os.path.dirname(instances_file)
    else:
        output_folder = args.output_dir

    if not os.path.exists(output_folder):
        os.makedirs(output_folder,exist_ok=True) #create folder if it does not exist
        print("Created output folder",output_folder)    

    #read the labels_file
    df_input_data = pd.read_csv(label_file) #read CSV label file using Pandas
    num_examples = len(df_input_data)
    if num_examples != 127:
        print("WARNING!!!! num_examples=", num_examples)

    #instances_file = r'justsounds_todoscorrigidos_normalizados\spectrogramD100T30.hdf5'
    #labels_dic_file = r'justsounds_todoscorrigidos_normalizados\labels_dictionary.json'
    X, y = read_instances_from_file(instances_file)
    y = y.astype(int) #convert output labels to integers
    #num_examples, D, T = X.shape
    if len(X.shape) == 3:
        num_examples, T, D = X.shape #we use the transpose
        num_rows = T
        num_columns = D
    else: #latent space
        num_examples, latent_dim = X.shape #we use the transpose
        num_rows = latent_dim
        num_columns = 1
    #print(num_examples, D, T)

    a_file = open(labels_dic_file, "r")
    label_dict_str = a_file.read()
    #https://appdividend.com/2022/01/29/how-to-convert-python-string-to-dictionary/
    label_dict = json.loads(label_dict_str)
    print("labels_dict", label_dict)
    print("labels_dict", type(label_dict))

    inv_label_dict = {v: k for k, v in label_dict.items()}
    class_names = []
    for i in range(len(label_dict.keys())):
        class_names.append(inv_label_dict[i])

    all_scores = list()
    #split into train and test correctly
    if use_disjoint_speakers:
        speaker_indices_dic, all_speakers = split_into_train_test_disjoint_speakers(df_input_data)
        num_speakers = len(all_speakers)
        for i in range(num_speakers):
            test_speaker = all_speakers[i]
            print("############## Evaluating test speaker", test_speaker)
            X_train, X_test, y_train, y_test, train_indices, test_indices = compose_train_test_disjoint_speakers(X, y, speaker_indices_dic, all_speakers, test_speaker)
            #read model for given speaker
            this_input_dnn_file_name = os.path.splitext(input_dnn_file_name)[0] + "_no_" + test_speaker + ".hdf5"
            error, confusion_matrix = test_trained_model(X_test, y_test, test_indices, num_rows, num_columns, this_input_dnn_file_name, df_input_data, label_dict)
            confusion_matrix_file = os.path.splitext(input_dnn_file_name)[0] + "_" + test_speaker + "_conf_matrix.png"
            save_confusion_matrix(confusion_matrix, confusion_matrix_file, class_names, test_speaker)
            all_scores.append(error)
        print(all_speakers)
        print('All error rates in %:')
        print(all_scores)
    else:
        #allow mixed speakers in train and test
        if test_indices_exist:
            temp_dir = os.path.dirname(instances_file)
            file_with_test_indices = os.path.join(temp_dir,"test_indices.csv")
            if not os.path.exists(file_with_test_indices):
                print("ERROR: could not find file", file_with_test_indices, "in folder", temp_dir)
                exit(-1)
            test_indices = np.genfromtxt(file_with_test_indices, delimiter=',')
            test_indices = test_indices.astype(int)
            file_with_train_indices = os.path.join(temp_dir,"train_indices.csv")
            train_indices = np.genfromtxt(file_with_train_indices, delimiter=',')
            train_indices = train_indices.astype(int)
            X_train = X[train_indices]
            y_train = y[train_indices]
            X_test = X[test_indices]
            y_test = y[test_indices]
        else:
            #be careful because these text examples will not be disjoint with the 
            #ones used in the training stage
            test_fraction = 0.2
            #print('y___',y)
            X_train, X_test, y_train, y_test, train_indices, test_indices = split_into_train_test_mixed_speakers(X, y, test_fraction)
        print("y_test, test_indices", y_test, test_indices)
        print("Test set has", y_test.shape[0],"examples")
        error, confusion_matrix = test_trained_model(X_test, y_test, test_indices, num_rows, num_columns, input_dnn_file_name, df_input_data, label_dict)
        confusion_matrix_file = os.path.splitext(input_dnn_file_name)[0] + "_mixed_speakers_conf_matrix.png"
        save_confusion_matrix(confusion_matrix, confusion_matrix_file, class_names, "mixed speakers")
