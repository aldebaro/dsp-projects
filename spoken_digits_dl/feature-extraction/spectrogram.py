'''
Deprecated. See utils_frontend.py
This code assumes Fs = 44100 Hz.
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
Assume Fs = 44100 Hz.
The STFT is calculated with dimension num_freq_bins x num_time_frames
but the code outputs a tranposed array with dimenstion
num_time_frames x num_freq_bins, which are called T and D in the code, respectively.
'''
def spectrogram(audio, D, T, normalize=True):
    samplerate = 44100
    spectrogram = []
    # STFT
    stft = librosa.stft(audio.astype(float),n_fft=2048, hop_length=None, win_length=None, window='hann', center=True, dtype=None, pad_mode='constant')
    #there is no energy above 15 kHz, so find the index
    df = samplerate/2048    
    kmax = int(15000/df)
    stft = stft[:kmax,:]
    samplerate = 30000 #update to plot correctly
    S_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    S_db = resize(S_db, (D, T))
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
    should_plot = False #plot spectrograms
    normalize= True #standardize each DFT frame of STFT
    if len(sys.argv) != 7:
        print("ERROR!")
        print("Usage: D T wav_folder label_file output_instances_file output_labels_dic_file")
        exit(1)
    D = int(sys.argv[1])
    T = int(sys.argv[2])
    wav_folder = sys.argv[3]
    label_file = sys.argv[4]
    output_instances_file = sys.argv[5]
    output_labels_dic_file = sys.argv[6]

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
            #recall that we save the transpose, so use transpose below to better plot
            s = X[i].T
            fig, ax = plt.subplots()
            #img = librosa.display.specshow(S_db,y_axis='log', x_axis='time', ax=ax,sr=samplerate)
            img = librosa.display.specshow(s, y_axis='linear', x_axis='time', ax=ax,sr=samplerate)
            ax.set_title('Power spectrogram')
            fig.colorbar(img, ax=ax, format="%+2.0f dB")
            #print('kk',img.shape)
            plt.savefig('books_read.png')
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
