import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import os
import numpy as np
import argparse
from ml_util import read_instances_from_file
import json

VERBOSE = 1 #use 0 or 1

def read_train_test_sets(instances_file):
            X, y = read_instances_from_file(instances_file)
            y = y.astype(int) #convert output labels to integers

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
            print("train_indices", train_indices, type(train_indices))
            print("test_indices", test_indices, type(test_indices))
            y_train = y[train_indices]
            y_test = y[test_indices]
            #X_test = [ X[i] for i in test_indices ]
            #X_train = [ X[i] for i in train_indices ]
            X_train = X[train_indices]
            X_test = X[test_indices]
            return X_train, y_train, X_test, y_test

if __name__ == '__main__':
    print("=====================================")
    print("Train and test classifiers")

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_indices_exist', help='a file called test_indices.csv in the input file folder must be used for defining the test set', action='store_true') #default is false
    #required arguments    
    parser.add_argument('--input_file',help='input file with features',required=True)
    parser.add_argument('--num_classes',type=int, help='number of classes (labels)', default=8)
    parser.add_argument('--epochs',type=int, help='epochs', default=500)

    args = parser.parse_args()
    num_classes = args.num_classes
    epochs = args.epochs

    #Force the user to acknowledge that we will use test_indices_exist in this code
    if not args.test_indices_exist:
        print("ERROR: the parameter --test_indices_exist must be used in the command line")
        exit(-1)

    instances_file = args.input_file

    # load the dataset
    # split into input (X) and output (y) variables
    X_train, y_train, X_test, y_test = read_train_test_sets(instances_file)

    input_dimension = X_train.shape[1]

    # define the keras model
    model = Sequential()
    model.add(Dense(10, input_shape=(input_dimension,), activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(num_classes, activation='sigmoid'))
    # compile the keras model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    out_folder = os.path.dirname(instances_file)

    best_model_name = os.path.join(out_folder, "best_model.hdf5")
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=best_model_name,
                save_weights_only=False,
                monitor='val_accuracy',
                mode='max',
                save_best_only=True)

    # fit the keras model on the dataset
    hist = model.fit(X_train, y_train, epochs=epochs, batch_size=3, verbose=VERBOSE,
            validation_split=0.4,
            callbacks=[model_checkpoint_callback])
    print("Saved best model", best_model_name)

    last_model_name = os.path.join(out_folder, "last_model.hdf5")
    model.save(last_model_name)
    print("Saved model", last_model_name)

    #print(hist.params)
    # Get the dictionary containing each metric and the loss for each epoch
    history_dict = hist.history
    # Save it under the form of a json file
    training_history_file_name = os.path.join(out_folder, "training_history_dic.json")
    json.dump(history_dict, open(training_history_file_name, 'w'))  
    print("Saved training history in file", training_history_file_name)

    #You can now access the value of the loss at the 50th epoch like this :
    #print(history_dict['loss'][49])
    #Reload it with
    #history_dict = json.load(open(training_history_file_name, 'r'))