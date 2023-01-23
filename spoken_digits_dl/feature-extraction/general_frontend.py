'''
This code uses the logic of spectrogram.py and the feature processing
of hyperresolution_magnasco_zoom.py.
While hyperresolution_magnasco_zoom.py writes spec and PNG files for each wav file,
this code writes a single HDF5 file for the whole set of wav files.
'''
import numpy as np
import pandas as pd
import os
import librosa
import librosa.display
import sys
import json
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import argparse
import pickle

#routines for feature extraction
from utils_frontend import *

if __name__ == '__main__':

    list_of_features = ["magnasco", "stft", "mel"]
    list_of_normalizers = ["minmax", "maggie", "std_freq", "none"]
    max_freq = 4000 #in Hz, for STFT and Mel features

    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_dir',help='input folder with wav files',default='../wav')
    parser.add_argument('--general_dir',help='folder with wavs_labels.csv and labels_dictionary.json files',default='../general')
    parser.add_argument('--output_file',help='output file for saving time-freq representation')
    parser.add_argument('--features',help='choose the features to be extracted',
        choices=list_of_features, default = "magnasco")
    parser.add_argument('--normalization',help='choose normalization method',
        choices=list_of_normalizers, default="minmax")
    parser.add_argument('--log_domain', action='store_true') #default is false
    parser.add_argument('--save_plots', action='store_true') #default is false
    parser.add_argument('--show_plot', action='store_true') #default is false
    #required arguments
    parser.add_argument('--D',help='number of frequency points',required=True)
    parser.add_argument('--T',help='number of time points',required=True)
    parser.add_argument('--output_dir',help='output folder',required=True)
    args = parser.parse_args()

    convert_to_log_domain = args.log_domain
    save_plots = args.save_plots
    normalization_method = args.normalization
    show_plot = args.show_plot #plot spectrograms
    general_folder = args.general_dir

    D = int(args.D)
    T = int(args.T)
    wav_folder = args.wav_dir
    features = args.features
    #label_file = args.label_file
    output_folder = args.output_dir
    if not os.path.exists(output_folder):
        os.makedirs(output_folder,exist_ok=True) #create folder if it does not exist
        print("Created output folder",output_folder)

    if args.output_file == None:
        #create a default name
        output_file = os.path.join(output_folder,"features_" + features + "D" + str(D) + "T" + str(T) + ".hdf5")
    else:
        output_file = args.output_file

    features_folder = os.path.join(os.path.splitext(output_file)[0],"features_no_resizing")
    os.makedirs(features_folder,exist_ok=True) #create folder if it does not exist

    if save_plots:        
        plots_output_folder = os.path.join(output_folder,"/pngs_")
        plots_output_folder = os.path.join(output_folder,os.path.basename(output_file))            
        plots_output_folder = os.path.splitext(plots_output_folder)[0]
        os.makedirs(plots_output_folder,exist_ok=True) #create folder if it does not exist
        print("Created output folder for plots in png format",plots_output_folder)

    if os.path.exists(output_file):
        print("ERROR: ", output_file, "already exists!")
        print("Delete this file or choose another name with command line parameter --output_file")
        exit(-1)

    #save this script in the output folder for reproducibility
    #if os.name == 'nt':
    #    command = "copy " + label_file + " " + output_folder
    #else: #Linux
    #    command = "cp " + label_file + " " + output_folder
    #os.system(command)
    #print("Executed: ", command)
    command_line_file = os.path.join(output_folder,'general_frontend_commandline_args.txt')
    with open(command_line_file, 'w') as f:
        f.write(' '.join(sys.argv[0:]))
        #f.write(str(sys.argv[0:]))
    print("Saved file", command_line_file)

    #define file names
    #output_instances_file = os.path.join(output_folder, output_file)
    label_file = os.path.join(general_folder,"wavs_labels.csv")
    labels_dic_file = os.path.join(general_folder,"labels_dictionary.json")

    #read input file
    df = pd.read_csv(label_file) #read CSV label file using Pandas
    num_examples = len(df)
    #read dictionary
    a_file = open(labels_dic_file, "r")
    label_dict_str = a_file.read()
    #https://appdividend.com/2022/01/29/how-to-convert-python-string-to-dictionary/
    label_dict = json.loads(label_dict_str)

    #initialize and pre-allocate space
    #X = np.zeros((num_examples, D, T))
    X = np.zeros((num_examples, T, D)) #note I am using the transpose
    list_all_labels = [] #initialize list

    min_freq_dimension = 1e30
    min_time_dimension = 1e30
    max_freq_dimension = -1e30
    max_time_dimension = -1e30
    for i, row in df.iterrows():        
        wavfile_basename = row['filename']
        wavfile = os.path.join(wav_folder, wavfile_basename)
        samplerate, audio = read(wavfile)

        if False: #enable for testing with simple signal of increasing frequency
            audio = librosa.chirp(sr=samplerate, fmin=110, fmax=22050, duration=1, linear=True)

        #now we can have files with Fs = 8 kHz, so disable the check below
        #if samplerate != 44100:
        #    raise Exception('Sample rate must be 44.1 kHz, but we found' + str(samplerate))
        if features == 'magnasco':
            spectrogram = magnasco_spectrogram(audio)
        elif features == 'stft':
            spectrogram = stft_spectrogram(audio, max_freq=max_freq)
        elif features == 'mel':
            spectrogram = melspectrogram(audio, max_freq=max_freq, n_mels=D) 

        this_D, this_T = spectrogram.shape
        if this_D > max_freq_dimension: max_freq_dimension = this_D
        if this_T > max_time_dimension: max_time_dimension = this_T
        if this_D < min_freq_dimension: min_freq_dimension = this_D
        if this_T < min_time_dimension: min_time_dimension = this_T

        if show_plot:
            print("Dimensions = ", spectrogram.shape)
            print("Range = ", np.min(spectrogram), np.max(spectrogram))
            plot_feature(spectrogram,"Initial features. " + features)
        if save_plots:
            if not show_plot:
                plot_feature_no_show(spectrogram,"Initial features. " + features)            
            plots_folder = os.path.join(plots_output_folder,"initial")
            os.makedirs(plots_folder,exist_ok=True) #create folder if it does not exist
            this_file_name = os.path.splitext(wavfile_basename)[0]  
            this_file_name = os.path.join(plots_folder, this_file_name + ".png")
            plt.savefig(this_file_name)
    
        if convert_to_log_domain:
            top_db = 60 #floor value in dB below the maximum
            spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max, top_db=top_db)
            if show_plot:
                print("Range in log = ", np.min(spectrogram), np.max(spectrogram))
                print("Shape of features = ", spectrogram.shape)
                plot_feature(spectrogram,"Features in log domain. " + features)
            if save_plots:
                if not show_plot:
                    plot_feature_no_show(spectrogram,"Features in log domain. " + features)
                plots_folder = os.path.join(plots_output_folder,"logdomain")
                os.makedirs(plots_folder,exist_ok=True) #create folder if it does not exist
                this_file_name = os.path.splitext(wavfile_basename)[0]  
                this_file_name = os.path.join(plots_folder, this_file_name + ".png")
                plt.savefig(this_file_name)
        if normalization_method != "none":
            if normalization_method == "maggie":
                spectrogram = normalize_as_maggie(spectrogram)
            if normalization_method == "minmax":
                spectrogram = normalize_to_min_max(spectrogram)
            if normalization_method == "std_freq":
                spectrogram = normalize_standardize_along_frequency(spectrogram)
            if show_plot:
                print("Range after normalization = ", np.min(spectrogram), np.max(spectrogram))
                print("Shape of features = ", spectrogram.shape)
                plot_feature(spectrogram,"Normalized features. " + features + " " + normalization_method)            
            if save_plots:
                if not show_plot:
                    plot_feature_no_show(spectrogram,"Normalized features. " + features)            
                plots_folder = os.path.join(plots_output_folder,"normalized")
                os.makedirs(plots_folder,exist_ok=True) #create folder if it does not exist
                this_file_name = os.path.splitext(wavfile_basename)[0]  
                this_file_name = os.path.join(plots_folder, this_file_name + ".png")
                plt.savefig(this_file_name)

        #before resizing, save features in individual files:
        output_features_before_resizing = os.path.join(features_folder, 
            os.path.splitext(wavfile_basename)[0] + "_fea.pkl")
        #save file
        #https://stackoverflow.com/questions/10592605/save-classifier-to-disk-in-scikit-learn
        with open(output_features_before_resizing, 'wb') as fid:
            pickle.dump(spectrogram, fid)

        S_db = resize(spectrogram, (D, T), preserve_range=True, anti_aliasing=False)
        if show_plot:
            print("Shape of features = ", S_db.shape)
            print("Range after resizing = ", np.min(S_db), np.max(S_db))
            plot_feature(S_db,"Resized final features. " + features + " " + normalization_method)        
        if save_plots:
            if not show_plot:
                plot_feature_no_show(S_db,"Resized final features. " + features)            
            plots_folder = os.path.join(plots_output_folder,"final")
            os.makedirs(plots_folder,exist_ok=True) #create folder if it does not exist
            this_file_name = os.path.splitext(wavfile_basename)[0]  
            this_file_name = os.path.join(plots_folder, this_file_name + ".png")
            plt.savefig(this_file_name)
        X[i] = S_db.T #use transpose because this is the expected by the neural nets
        label = str(row['annotation']).strip() #column annotation indicates the labels
        list_all_labels.append(label)

        print(label,"from", os.path.basename(wavfile),"converted to", output_features_before_resizing)

    #now create the y array with label indices (not the strings)
    y = np.zeros((num_examples),dtype=int)
    for i in range(num_examples):
        this_label = list_all_labels[i]
        label_index = label_dict[this_label]
        y[i] = label_index

    write_instances_to_file(output_file, X, y)
    print("Wrote file", output_file)
    if False: #if wants to double check
        print('y1',y)
        X, y = read_instances_from_file(output_file)
        print('y2',y)

    print("Considering original features")
    print("Frequency dimension range = ",min_freq_dimension, max_freq_dimension)
    print("Time dimension range = ",min_time_dimension, max_time_dimension)
