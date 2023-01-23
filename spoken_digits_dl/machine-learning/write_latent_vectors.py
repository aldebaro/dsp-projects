'''
Converts to latent space
'''
import sys
import os
import numpy as np
from tensorflow.keras.models import load_model
from ml_util import read_instances_from_file
from ml_util import write_instances_to_file
import argparse
import pandas as pd
import json
'''
Write instances as vectors in latent space.
'''
def write_latent_vectors(encoder, X, y, name, out_folder):
    X_latent = encoder.predict(X)
    output_file = os.path.join(out_folder, name)
    write_instances_to_file(output_file, X_latent, y)
    print('X_latent.shape', X_latent.shape)
    num_examples, K = X_latent.shape #we use the transpose
    print("Wrote file", output_file, 'with', num_examples, 'examples of dimension',K)
    #write txt file
    txt_file = os.path.splitext(output_file)[0] + ".csv"
    #np.savetxt(txt_file, X_latent, header=output_file, delimiter=',')
    #print("also wrote CSV file with header", txt_file)
    np.savetxt(txt_file, X_latent, delimiter=',')
    print("also wrote CSV file without header", txt_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--general_dir',help='folder with wavs_labels.csv and labels_dictionary.json files',default='../general')
    parser.add_argument('--output_dir',help='output folder (default is the folder of the input file)')
    #required arguments    
    parser.add_argument('--input_file',help='input file with features',required=True)
    parser.add_argument('--input_model',help='input file with ML model (or prefix when using --disjoint_speakers)',required=True)
    args = parser.parse_args()

    instances_file = args.input_file
    #general_dir = args.general_dir
    #label_file = os.path.join(general_dir, "wavs_labels.csv")
    #labels_dic_file = os.path.join(general_dir, "labels_dictionary.json")
    encoder_file = args.input_model #file with DNN model

    if args.output_dir == None:
        #create a default name
        #output_folder = os.path.dirname(instances_file)
        output_folder = os.path.dirname(encoder_file)
    else:
        output_folder = args.output_dir

    if not os.path.exists(output_folder):
        os.makedirs(output_folder,exist_ok=True) #create folder if it does not exist
        print("Created output folder",output_folder)    

    #instances_file = r'justsounds_todoscorrigidos_normalizados\spectrogramD100T30.hdf5'
    #labels_dic_file = r'justsounds_todoscorrigidos_normalizados\labels_dictionary.json'
    X, y = read_instances_from_file(instances_file)
    y = y.astype(int) #convert output labels to integers
    #num_examples, D, T = X.shape
    num_examples, T, D = X.shape #we use the transpose
    #print(num_examples, D, T)
    num_rows = T
    num_columns = D

    X, y = read_instances_from_file(instances_file)
    y = y.astype(int) #convert output labels to integers
    #num_examples, D, T = X.shape
    num_examples, T, D = X.shape #we use the transpose

    enc = load_model(encoder_file)

    write_latent_vectors(enc, X, y, "latent_vectors.h5", output_folder)
