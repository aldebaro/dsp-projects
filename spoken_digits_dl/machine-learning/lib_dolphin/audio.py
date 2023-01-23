from importlib.resources import path
import random
import numpy as np
import pandas as pd
import os
#from numba import jit
import librosa
import librosa.display

from numpy.fft        import fft
from scipy.io.wavfile import read, write    
from skimage.transform import resize

def raw(path):
    try:
        x = read(path)[1]
        if len(x.shape) > 1:
            return x[:, 0]
        return x
    except:
        print("Could not read file: {}".format(path))
        return np.zeros(0)
    
def old_spectrogram(audio, lo = 20, hi = 200, win = 512, step=128, normalize=True):
    spectrogram = []
    hanning = np.hanning(win)
    for i in range(win, len(audio), step):
        dft = np.abs(fft(audio[i - win: i] * hanning))
        if normalize:
            mu  = np.mean(dft)
            std = np.std(dft) + 1.0 # @TODO this +1 is suspicious
            spectrogram.append((dft - mu) / std)
        else:
            spectrogram.append(dft)        
    spectrogram = np.array(spectrogram)
    print('AA', spectrogram.shape)
    spectrogram = spectrogram[:, win//2:][:, lo:hi]
    print('BB', spectrogram.shape)
    return spectrogram

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
    for i in range(num_time_frames):
        dft = S_db[:,i]
        if normalize:
            mu  = np.mean(dft)
            #std = np.std(dft) + 1.0 # @TODO this +1 is suspicious
            std = np.std(dft)
            if std > 0:
                spectrogram.append((dft - mu) / std)
            else:
                spectrogram.append((dft - mu))
        else:
            spectrogram.append(dft)        
    spectrogram = np.array(spectrogram)
    #print('spectrogram.shape', spectrogram.shape)
    return spectrogram

def resolve_window(sample_raw, window, step, fwd = True):
    if fwd:
        return int((sample_raw - window) / step)
    else:
        return int(sample_raw * step + window)

    
#@jit
def windowing(region, window):
    N, D = region.shape
    windows = []
    if N > window:
        step = window // 2
        for i in range(window, N, step):
            r = region[i - window:i].reshape((window, D, 1))
            windows.append(r)
        return np.stack(windows)
    else:
        return None
        

def dataset_unsupervised_regions_windowed(regions, wavfile, encoder, supervised, label_dict, lo, hi, win, step, T, l2_window, dont_window_whistle):
    df        = pd.read_csv(regions)
    N         = len(df)
    audio     = raw(wavfile) 
    instances = []
    labels    = []
    ids       = []
    for i, row in df.iterrows():
        start = row['starts']
        stop  = row['stops']
        w     = audio[start:stop]
        if len(w) > 0:
            s = spectrogram(w, lo, hi, win, step)
            w = windowing(s, T)
            if w is not None:
                if i % 100 == 0:
                    print(" ... reading {}/{}={}".format(i, N, i / N))
                x = encoder.predict(w)
                y = supervised.predict(w)

                n_wstl = 0
                n_others = 0
                for l in np.argmax(y, axis=1):
                    l = label_dict[l]
                    if l == 'WSTL_UP' or l == 'WSTL_DOWN':
                        n_wstl += 1
                    else:
                        n_others += 1
                print("FRAMES: {} / {}".format(n_wstl, n_others))
                if n_wstl > n_others and dont_window_whistle:
                    print(".. whole")
                    instances.append(x)
                    labels.append(y)
                    ids.append(i)
                else:
                    print(".. window")
                    # TODO add windowing info for export visuals
                    for j in range(l2_window, len(x), l2_window // 2):
                        instances.append(x[j - l2_window:j])
                        labels.append(y[j - l2_window:j])
                        ids.append(i)
    return ids, instances, labels

        
def dataset_unsupervised_regions(regions, wavfile, encoder, supervised, lo, hi, win, step, T):
    df        = pd.read_csv(regions)
    N         = len(df)
    audio     = raw(wavfile) 
    instances = []
    labels    = []
    ids       = []
    for i, row in df.iterrows():
        start = row['starts']
        stop  = row['stops']
        w     = audio[start:stop]
        if len(w) > 0:
            s = spectrogram(w, lo, hi, win, step)
            w = windowing(s, T)
            if w is not None:
                if i % 100 == 0:
                    print(" ... reading {}/{}={}".format(i, N, i / N))
                x = encoder.predict(w)
                y = supervised.predict(w)                
                instances.append(x)
                labels.append(y)
                ids.append(i)
    return ids, instances, labels


def dataset_unsupervised_windows(label, wavfile, lo, hi, win, step, raw_size, T, n = 10000):
    df = pd.read_csv(label)
    audio     = raw(wavfile)
    instances = []
    for _, row in df.iterrows():
        start = row['starts']
        stop  = row['stops']
        w     = audio[start:stop]
        if len(w) > 0:
            s = spectrogram(w, lo, hi, win, step)
            w = windowing(s, T)
            if w is not None:
                for i in range(0, len(w)):
                    instances.append(w[i])
    random.shuffle(instances)
    return instances[0:n]
    

def dataset_supervised_windows(label, wavfile, lo, hi, win, step, raw_size):
    df        = pd.read_csv(label)
    #print(df)
    #print('raw_size',raw_size)
    audio     = raw(wavfile)
    print("DIFF: {}".format(len(audio) - (df['offset'].max() + raw_size)))
    labels    = []
    instances = []
    ra = []
    label_dict = {}
    cur_label  = 0
    for _, row in df.iterrows():
        start = row['offset']
        stop  = start + raw_size
        #print(start,stop)
        label = row['annotation'].strip()
        if label not in label_dict:
            label_dict[label] = cur_label
            cur_label += 1
        w = audio[start:stop]
        #print(w, lo, hi, win, step)

        s = spectrogram(w, lo, hi, win, step)
        f, t = s.shape
        ra.append(w)
        instances.append(s)
        labels.append(label_dict[label])
    return instances, ra, labels, label_dict

#n seems the maximum number
def ak_dataset_unsupervised_windows(label, wavfolder, D, T, n = 10000):
    df = pd.read_csv(label)
    #audio     = raw(wavfile)
    instances = []
    for _, row in df.iterrows():
        wavfile = os.path.join(wavfolder,row['filename'])
        print('wavfile', wavfile)
        audio     = raw(wavfile)
        start = row['starts']
        stop  = row['stops']
        w     = audio[start:stop]
        if len(w) > 0:
            #s = spectrogram(w, lo, hi, win, step)
            s = spectrogram(w, D, T)
            print('s', s.shape)
            #w = windowing(s, T) #suspicious
            #print('w', w.shape)
            w = s
            if w is not None:
                for i in range(0, len(w)):
                    instances.append(w[i])
    random.shuffle(instances)
    return instances[0:n]

'''
Returns:
instances => list of spectrograms of dimension F x T
ra => raw audio, used to calculate spectrograms 
labels => list of labels
label_dict => labels dictionary
'''
def ak_dataset_supervised_windows(label, wavfolder, D, T):
    df        = pd.read_csv(label) #read CSV label file using Pandas
    #print(df)
    #print('raw_size',raw_size)
    #print("DIFF: {}".format(len(audio) - (df['offset'].max() + raw_size)))
    labels    = []
    instances = []
    ra = []
    label_dict = {}
    cur_label  = 0
    for _, row in df.iterrows():        
        wavfile = os.path.join(wavfolder,row['filename'])
        audio     = raw(wavfile)
        #start = row['offset']
        start = row['starts']
        stop  = row['stops']
        #stop  = start + raw_size
        #print(start,stop)
        label = row['annotation'].strip() #column annotation indicates the labels
        if label not in label_dict:
            label_dict[label] = cur_label
            cur_label += 1
        w = audio[start:stop]
        #print(w, lo, hi, win, step)

        s = spectrogram(w, D, T)
        f, t = s.shape #see Fig. 2 of Daniel's paper: spectrogram has dimension T x F
        #print('f,t=',f,t)
        ra.append(w)
        instances.append(s)
        labels.append(label_dict[label])
    return instances, ra, labels, label_dict

def ak_dataset_unsupervised_regions(regions, wavfolder, encoder, supervised, lo, hi, win, step, T):
    df        = pd.read_csv(regions)
    N         = len(df)
    #audio     = raw(wavfile) 
    instances = []
    labels    = []
    ids       = []
    for i, row in df.iterrows():
    #for _, row in df.iterrows():        
        wavfile = os.path.join(wavfolder,row['filename'])
        audio     = raw(wavfile)
        start = row['starts']
        stop  = row['stops']
        w     = audio[start:stop]
        if len(w) > 0:
            s = spectrogram(w, lo, hi, win, step)
            w = windowing(s, T)
            if w is not None:
                if i % 100 == 0:
                    print(" ... reading {}/{}={}".format(i, N, i / N))
                x = encoder.predict(w)
                y = supervised.predict(w)                
                instances.append(x)
                labels.append(y)
                ids.append(i)
    return ids, instances, labels

def ak_dataset_unsupervised_regions_windowed(regions, wavfolder, encoder, supervised, label_dict, lo, hi, win, step, T, l2_window, dont_window_whistle):
    df        = pd.read_csv(regions)
    N         = len(df)
    #audio     = raw(wavfile) 
    instances = []
    labels    = []
    ids       = []
    for i, row in df.iterrows():
        wavfile = os.path.join(wavfolder,row['filename'])
        audio     = raw(wavfile)
        start = row['starts']
        stop  = row['stops']
        w     = audio[start:stop]
        if len(w) > 0:
            s = spectrogram(w, lo, hi, win, step)
            w = windowing(s, T)
            if w is not None:
                if i % 100 == 0:
                    print(" ... reading {}/{}={}".format(i, N, i / N))
                x = encoder.predict(w)
                y = supervised.predict(w)

                n_wstl = 0
                n_others = 0
                for l in np.argmax(y, axis=1):
                    l = label_dict[l]
                    if l == 'WSTL_UP' or l == 'WSTL_DOWN':
                        n_wstl += 1
                    else:
                        n_others += 1
                print("FRAMES: {} / {}".format(n_wstl, n_others))
                if n_wstl > n_others and dont_window_whistle:
                    print(".. whole")
                    instances.append(x)
                    labels.append(y)
                    ids.append(i)
                else:
                    print(".. window")
                    # TODO add windowing info for export visuals
                    for j in range(l2_window, len(x), l2_window // 2):
                        instances.append(x[j - l2_window:j])
                        labels.append(y[j - l2_window:j])
                        ids.append(i)
    return ids, instances, labels