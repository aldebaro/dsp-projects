import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import pandas as pd
import os

from collections import Counter
from collections import namedtuple
from lib_dolphin.audio import *
from lib_dolphin.htk_helpers import * 
from scipy.io.wavfile import read, write    

from matplotlib.colors import Normalize
import os

#TODO get rid of this need of reading colors.txt
color_file = 'lib_dolphin/colors.txt'
if not os.path.exists(color_file):
    color_file = 'machine-learning/lib_dolphin/colors.txt'
if not os.path.exists(color_file):
    print("ERROR: Script needs to find the file colors.txt.")    
    print("You should run it from a folder that has a subfolder lib_dolphin or a subfolder machine-learning/lib_dolphin")
    exit(-1)
COLORS = list(
    pd.read_csv(color_file, sep='\t', header=None)[1].apply(lambda x: x + "80")
)


BIAS   = 0.7
START  = 0.2
STOP   = 0.9
SCALER = 1.0


def plot_annotations(anno_files, labels, wav_folder, out_folder, win, th, noise_th = 0.9, plot_noise = True, do_compress=False):
    n = -1
    filtered = {}
    anno_mapping = {}
    cur = 0 
    for file, annotations in anno_files.items():
        n += 1
        if len(annotations) > 1 and do_compress:
            annotations = compress(annotations)
        if len(annotations) > 1:
            filtered[file] = []
            path = "{}/{}.wav".format(wav_folder, file)
            x = raw(path)
            s = spectrogram(x, lo = 0, hi = 256)

            max_annotations = max([stop for _, stop, _, _ in annotations])
            lab_df = pd.read_csv(labels[file])
            print(file, n, len(s), len(lab_df), max_annotations)
            if len(s) < 10000:
                fig, ax = plt.subplots()
                fig.set_size_inches(len(s) / 100, len(s[0]) / 100)
                ax.imshow(BIAS - s.T * SCALER, norm=Normalize(START, STOP), cmap='gray')
                for start, stop, i, ll in annotations:
                    label_regions = list(lab_df['labels'][start:stop])
                    probs         = list(lab_df['prob'][start:stop])
                    label_regions = [label_regions[i] for i in range(0, len(label_regions))]
                    counter = Counter(label_regions)
                    n_noise = counter['NOISE'] 
                    n_not_noise = len(label_regions) - n_noise
                    if n_noise + n_not_noise == 0:
                        ratio = 0.0
                    else:
                        ratio = n_noise / (n_not_noise + n_noise)
                    is_ll    = ll <= th
                    is_noise = ratio >= noise_th
                    is_sil   = i == 'sil'
                    supress  = is_sil or is_noise or is_ll
                    print("SUPRESS: {}, {} < {}, {} > {}, sil {}".format(supress, ll, th, ratio, noise_th, is_sil))
                    if not supress or plot_noise:                    
                        if not supress:
                            filtered[file].append((start, stop, i, ll))                        
                        a = start * win
                        e = stop  * win
                        if not supress: 
                            if i not in anno_mapping:
                                anno_mapping[i] = cur
                                cur += 1
                            plt.text(a + (e - a) // 2 , 30, i, size=20)
                            rect = patches.Rectangle((a, 0), e - a, 256, linewidth=1, edgecolor='r', facecolor=COLORS[anno_mapping[i] + 1])
                            ax.add_patch(rect)
                        if is_noise and supress:
                            print("Neural")
                            plt.text(a, 30, "N", size=20)
                        if is_ll and supress:
                            print("Likelihood")
                            plt.text(a, 50, "L", size=20)
                        if is_sil and supress:
                            print("HTK")
                            plt.text(a, 70, "H", size=20)                            

                plt.savefig("{}/{}.png".format(out_folder, file))
                plt.close()
            else:
                print("\t skip")
    return filtered


def plot_result_matrix(confusion, classes, predictions, title, cmap=plt.cm.Blues):
    fig, ax = plt.subplots()
    fig.set_size_inches(20, 20)
    im = ax.imshow(confusion, interpolation='nearest', cmap=cmap)
    ax.set(xticks=np.arange(confusion.shape[1]),
           yticks=np.arange(confusion.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=predictions, yticklabels=classes,
           title=title,
           ylabel='True Label',
           xlabel='Predicted Label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    fmt = '.1f'
    thresh = confusion.max() / 2.
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            ax.text(j, i, format(confusion[i, j], fmt),
                    ha="center", va="center",
                    color="white" if confusion[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def plot_tensorflow_hist(hist, output):
    if 'mse' in hist.history:
        plt.plot(hist.history['mse'],     label="mse")
        plt.plot(hist.history['val_mse'], label="val_mse")
        plt.title("MSE")
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('mse')
    elif 'accuracy' in hist.history:
        plt.plot(hist.history['accuracy'],     label="accuracy")
        plt.plot(hist.history['val_accuracy'], label="val_accuracy")
        plt.title("Accuracy")
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
    plt.savefig(output)
    plt.clf()


def visualize_dataset(instances, output, D, T):
    plt.figure(figsize=(20, 20))
    for i in range(0, 100):
        xi = np.random.randint(0, len(instances))
        plt.subplot(10, 10, i + 1)
        #D=20 and T=30
        plt.imshow(1.0 - instances[xi].reshape((D, T)).T, cmap='gray')
        #plt.imshow(1.0 - instances[xi].reshape((36, 130)).T, cmap='gray')
    plt.savefig(output)
    plt.clf()

    
def enc_filters(enc, n_filters, n_banks, output):    
    d = 1
    plt.figure(figsize=(10, 10))
    for s in range(0, 2 * n_banks, 2):
        w = enc.weights[s].numpy()
        for i in range(w.shape[-1]):
            weight = w[:, :, 0, i]
            x, y = weight.shape
            plt.subplot(n_filters // 8, 8, d)
            if x > y:
                fig = plt.imshow(weight.T, cmap='gray')
            else: 
                fig = plt.imshow(weight, cmap='gray')
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            d += 1
    plt.savefig(output)
    plt.clf()
