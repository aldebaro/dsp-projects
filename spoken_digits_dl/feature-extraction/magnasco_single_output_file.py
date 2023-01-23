'''
Deprecated. See utils_frontend.py

This code uses the logic of spectrogram.py and the feature processing
of hyperresolution_magnasco_zoom.py.
While hyperresolution_magnasco_zoom.py writes spec and PNG files for each wav file,
this code writes a single HDF5 file for the whole set of wav files.
'''
import numpy as np
import pandas as pd
import os
#from numba import jit
import librosa
import librosa.display
import sys
import h5py
import json
from numpy.fft        import fft
from scipy.io.wavfile import read, write    
from skimage.transform import resize
import matplotlib.pyplot as plt
import torch
import reassignment.reassignment_linear as reassign_lin
#from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler

should_plot = False #plot Magnasco spectrograms
normalize= False #standardize each DFT frame of the Magnasco representation
normalize_as_maggie = False #Maggie moves all values to 0 and 1, and sum 1 to all values, such that several lines are shows

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
Use Magnasco algorithm with reassign_lin.high_resolution_spectrogram.
'''
def spectrogram(audio, D, T, normalize=False):
    # defining constants and parameters
    q = 2
    tdeci = 96
    over = 2
    noct = 108
    minf = 1.5e-2
    maxf = 0.16

    spectrogram = reassign_lin.high_resolution_spectrogram(audio, q, tdeci, over, noct, minf, maxf, device=torch.device('cpu'))
    # port the spectrogram to cpu
    # No need to do it if device = cpu
    spectrogram = spectrogram.cpu().numpy().T

    if normalize_as_maggie:
        # Original author: Jiayi (Maggie) Zhang <jiayizha@andrew.cmu.edu>
        spectrogram = np.log(spectrogram+1e-8)
        spectrogram /= np.max(np.abs(spectrogram))
        spectrogram += 1
    else:
        #use librosa
        spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)
        #print('aaaaaa', np.max(spectrogram), np.min(spectrogram))
        #numbers are now from -80 dB to 0 dB. Move them to range -1 to 1
        # Using sklearn MinMaxScaler, note it works for each column, so
        # reshape to make it a single column array
        min_max_scaler = MinMaxScaler(feature_range=(-1.0,1.0))    
        # Stack everything into a single column to scale by the global min / max
        original_shape = spectrogram.shape
        tmp = spectrogram.reshape(-1,1)
        spectrogram = min_max_scaler.fit_transform(tmp).reshape(original_shape)
        #print('bbbb', np.max(spectrogram), np.min(spectrogram))
    
    if should_plot:
            #recall that it is already using the transpose, so do not transpose
            #s = spectrogram.T
            s = spectrogram
            fig, ax = plt.subplots()
            #img = librosa.display.specshow(S_db,y_axis='log', x_axis='time', ax=ax,sr=samplerate)
            img = librosa.display.specshow(s, y_axis='linear', x_axis='time', ax=ax,sr=samplerate)
            ax.set_title('Power spectrogram before resize')
            fig.colorbar(img, ax=ax, format="%+2.0f dB")
            #print('kk',img.shape)
            plt.savefig('can_delete_later.png')
            plt.show()

    #S_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    S_db = resize(spectrogram, (D, T))
    #S_db = resize(spectrogram, (T, D)) #because it is transposed
    #S_db = S_db.T #note we assume it is not transposed below
    #print('S_db.shape', S_db.shape)
    num_freq_bins, num_time_frames = S_db.shape
    if normalize:
        for i in range(num_time_frames):
            dft = S_db[:,i]
            mu  = np.mean(dft)
            #std = np.std(dft) + 1.0 # @TODO this +1 is suspicious
            std = np.std(dft)
            if std > 0:
                S_db[:,i] = (dft - mu) / std
            else:
                S_db[:,i] = (dft - mu)
    #print('S_db',S_db)
    S_db = S_db.T #note we use the transpose
    return S_db

if __name__ == '__main__':
    if len(sys.argv) != 7:
        print("ERROR!")
        print("Usage: D T wav_folder label_file output_instances_file output_labels_dic_file")
        exit(1)
    D = int(sys.argv[1])
    T = int(sys.argv[2])
    wav_folder = sys.argv[3]
    label_file = sys.argv[4]
    output_instances_file = sys.argv[5]
    output_folder = os.path.dirname(output_instances_file)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder,exist_ok=True) #create folder if it does not exist
        print("Created output folder",output_folder)
    #save this script in the output folder for reproducibility
    if os.name == 'nt':
        command = "copy " + label_file + " " + output_folder
    else: #Linux
        command = "cp " + label_file + " " + output_folder
    os.system(command)
    print("Executed: ", command)

    output_labels_dic_file = sys.argv[6]
    if not os.path.exists(output_labels_dic_file):
        os.makedirs(output_folder,exist_ok=True) #create folder if it does not exist
        print("Created output folder",output_folder)

    df = pd.read_csv(label_file) #read CSV label file using Pandas
    num_examples = len(df)

    #initialize and pre-allocate space
    label_dict = {} #dictionary
    cur_label = 0 #current label index
    #X = np.zeros((num_examples, D, T))
    X = np.zeros((num_examples, T, D)) #note I am using the transpose
    list_all_labels = [] #initialize list
    for i, row in df.iterrows():        
        wavfile = os.path.join(wav_folder,row['filename'])
        samplerate, audio = read(wavfile)
        if samplerate != 44100:
            raise Exception('Sample rate must be 44.1 kHz, but we found' + str(samplerate))
        log_spectrogram = spectrogram(audio, D, T, normalize=normalize)
        X[i] = log_spectrogram
        if should_plot:
            #recall that we use the transpose
            s = X[i].T
            fig, ax = plt.subplots()
            #img = librosa.display.specshow(S_db,y_axis='log', x_axis='time', ax=ax,sr=samplerate)
            img = librosa.display.specshow(s, y_axis='linear', x_axis='time', ax=ax,sr=samplerate)
            ax.set_title('Power spectrogram of final features')
            fig.colorbar(img, ax=ax, format="%+2.0f dB")
            #print('kk',img.shape)
            plt.savefig('can_delete_later_2.png')
            #print('cccc', np.max(s), np.min(s))
            plt.show()

        label = row['annotation'].strip() #column annotation indicates the labels
        list_all_labels.append(label)
        if label not in label_dict:
            label_dict[label] = cur_label
            cur_label += 1

        print(label,"from", os.path.basename(wavfile))                

    #now create the y array with label indices (not the strings)
    y = np.zeros((num_examples),dtype=int)
    for i in range(num_examples):
        this_label = list_all_labels[i]
        label_index = label_dict[this_label]
        y[i] = label_index

    write_instances_to_file(output_instances_file, X, y)
    if False: #if wants to double check
        print('y1',y)
        X, y = read_instances_from_file(output_file)
        print('y2',y)

    a_file = open(output_labels_dic_file, "w")
    json.dump(label_dict, a_file)
    a_file.close()
    print('Wrote instances file',output_instances_file,'with D =',D,'T =',T)
    print('Wrote labels dictionary', output_labels_dic_file)
