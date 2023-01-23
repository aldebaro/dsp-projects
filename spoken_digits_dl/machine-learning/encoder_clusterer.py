'''
From paper "An Auto Encoder For Audio Dolphin Communication" by
Daniel Kohlsdorf, dkohlsdorf@gmail.com
Denise Herzing, dlherzing@wilddolphinproject.org 
and Thad Starner College of Computing, Georgia Institute of Technology
Part of the "Wild Dolphin Project"
'''
import json
import numpy as np
import pickle as pkl
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse

from sklearn.model_selection import train_test_split
import random

import nmslib

#from pipeline import train_triplets
from lib_dolphin.audio import *
from lib_dolphin.features import *
from lib_dolphin.features import TripletLoss
from lib_dolphin.eval import *
from lib_dolphin.dtw import *
from lib_dolphin.htk_helpers import *
from lib_dolphin.sequential import *
from lib_dolphin.statistics import * 

from ml_util import *
#from ml_util import read_instances_from_file

from collections import namedtuple, Counter, defaultdict

from scipy.io.wavfile import read, write
from scipy.spatial import distance
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import *
from sklearn.cluster import AgglomerativeClustering, KMeans
from kneed import KneeLocator

from subprocess import check_output

#NEURAL_NOISE_DAMPENING=0.5
#NEURAL_SMOOTH_WIN=64
#NEURAL_SIZE_TH=32

#FFT_STEP     = 128
#FFT_WIN      = 512
#FFT_HI       = 230
#FFT_LO       = 100

#D            = FFT_WIN // 2 - FFT_LO - (FFT_WIN // 2 - FFT_HI)
#RAW_AUDIO    = 5120
#T            = int((RAW_AUDIO - FFT_WIN) / FFT_STEP)

#D = 100
#T = 30

CONV_PARAM   = [
    (8, 8,  32),
    (4, 16, 32),
    (2, 32, 32),
    (1, 64, 32),
    (8,  4, 32),
    (16, 4, 32),
    (32, 4, 32)
]

N_BANKS = len(CONV_PARAM)
N_FILTERS = int(np.sum([i for _, _, i in CONV_PARAM]))

'''
Originally in file eval.py
'''
def visualize_dataset(instances, output, D, T):
    plt.figure(figsize=(20, 20))
    for i in range(0, 100):
        xi = np.random.randint(0, len(instances))
        plt.subplot(10, 10, i + 1)
        plt.imshow(1.0 - instances[xi].reshape((D, T)).T, cmap='gray')
        #plt.imshow(1.0 - instances[xi].reshape((36, 130)).T, cmap='gray')
    plt.savefig(output)
    plt.clf()

'''
Performs K-means clustering using sklearn.
'''
def cluster_model(data, out_folder, label, min_k=2, max_k=26): 
    if max_k is None:
        km = KMeans(n_clusters=min_k)
        km.fit(data)
        return km
    
    scores = []
    models = []
    for k in range(min_k, max_k):
        km = KMeans(n_clusters=k)
        km.fit(data)
        bic = compute_bic(km, data)
        scores.append(bic)
        models.append(km)
    #some discussion about knee locator:
    #https://datascience.stackexchange.com/questions/6508/k-means-incoherent-behaviour-choosing-k-with-elbow-method-bic-variance-explain
    kn = KneeLocator(np.arange(len(scores)), scores, curve='concave', direction='increasing')
    model = models[kn.knee]
    plt.plot([km.n_clusters for km in models], scores)
    plt.vlines(model.n_clusters, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
    plt.title(f'Knee at {model.n_clusters}')
    plt.savefig(f'{out_folder}/{label}_cluster_knee.png')
    plt.close()
    return model

'''
Write instances as vectors in latent space.
'''
def write_latent_vectors(encoder, X, y, label_dict, name, out_folder):
    X_latent = encoder.predict(X)

'''
Search using K nearest-neighbor (K-NN). Assume y_test and y_train are integers.
'''
def neighbours_encoder(encoder, x_train, y_train, x_test, y_test, label_dict, name, out_folder, BATCH=10):
    x_train = encoder.predict(x_train, batch_size = BATCH)
    x_test = encoder.predict(x_test, batch_size = BATCH)

    # Non-Metric Space Library (NMSLIB) is an efficient cross-platform similarity
    # search library and a toolkit for evaluation of similarity search methods. 
    index = nmslib.init(method='hnsw', space='cosinesimil')
    index.addDataPointBatch(x_train)
    index.createIndex({'post': 2}, print_progress=True)
    #see https://benfred.github.io/nmslib/quickstart.html
    neighbours = index.knnQueryBatch(x_test, k=10, num_threads=4)

    n = len(label_dict)
    label_names = ["" for i in range(n)]
    for l, i in label_dict.items():
        label_names[i] = l
    confusion = np.zeros((n,n))

    for i, (ids, _) in enumerate(neighbours):
        labels = [int(y_train[i]) for i in ids]
        c      = Counter(labels)
        l      = [(k, v) for k, v in c.items()]
        l      = sorted(l, key = lambda x: x[1], reverse=True)[0][0]
        confusion[y_test[i], l] += 1

    accuracy = np.sum(confusion * np.eye(n)) / len(y_test)
    plot_result_matrix(confusion, label_names, label_names, "confusion {} {}".format(name, accuracy))
    plt.savefig('{}/confusion_nn_{}.png'.format(out_folder, name))
    plt.close()
    return accuracy

#def triplets(by_label, n = 50000):
def triplets(by_label, n = 5):
    '''
    Returns a list with n triplets (anchor, positive, negative) examples.
    Discussion by Daniel about triplet loss for siamese neural network architectures:
    http://daniel-at-world.blogspot.com/2019/07/implementing-triplet-loss-function-in.html
    '''
    l = list(by_label.keys())
    #N = len(l)
    #print("N=",N)
    #print('by_label',by_label)
    for i in range(n):
        pos_label = np.random.randint(0, len(l))
        neg_label = np.random.randint(0, len(l))
        while neg_label == pos_label:
            neg_label = np.random.randint(0, len(l))
        #print("pos_label,neg_label=",pos_label,neg_label)
        #the anchor, positive and negative examples            
        anc_i     = np.random.randint(0, len(by_label[pos_label]))
        pos_i     = np.random.randint(0, len(by_label[pos_label]))
        while anc_i == pos_i:
            pos_i     = np.random.randint(0, len(by_label[pos_label]))
        neg_i     = np.random.randint(0, len(by_label[neg_label]))
        #print('anc_i,pos_i,neg_i=',anc_i,pos_i,neg_i)
        #yield is a keyword that is used like return, except the function will return a generator.
        #https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do
        #print('####', by_label[pos_label][anc_i], by_label[pos_label][pos_i], by_label[neg_label][neg_i])
        yield by_label[pos_label][anc_i], by_label[pos_label][pos_i], by_label[neg_label][neg_i]

'''
triplet_model is a method in features.py
Discussion by Daniel about triplet loss for siamese neural network architectures:
http://daniel-at-world.blogspot.com/2019/07/implementing-triplet-loss-function-in.html
'''            
def train_triplets(enc, by_label, WINDOW_PARAM, num_random_triplets, epochs=10, latent_dim = 128, batch=3):
    model = triplet_model(WINDOW_PARAM, enc, latent_dim) #create the model based on the given encoder
    print("Siamese model")
    print(model.summary())
    for epoch in range(epochs):
        batch_pos = []
        batch_neg = []
        batch_anc = []
        n = 0
        total_loss = 0.0
        for anc, pos, neg in triplets(by_label,n=num_random_triplets):
            #print('anc, pos, neg',anc, pos, neg)
            batch_pos.append(pos)
            batch_neg.append(neg)
            batch_anc.append(anc)
            if len(batch_pos) == batch:
                batch_anc = np.stack(batch_anc)
                batch_pos = np.stack(batch_pos)
                batch_neg = np.stack(batch_neg)
                #note that all output labels are zero
                #https://stackoverflow.com/questions/62629997/difference-between-tensorflow-model-fit-and-train-on-batch
                #loss = model.train_on_batch(x=[batch_anc, batch_pos, batch_neg], y=np.zeros((BATCH,  256)))
                z=[batch_anc, batch_pos, batch_neg]
                #print('@@@@@@@@ z=', z[0].shape, z[1].shape, z[2].shape)
                #print('z',np.unique(z))
                #x is a list of 3 numpy arrays
                output_dimension =  3 * latent_dim #dimension of y
                loss = model.train_on_batch(x=[batch_anc, batch_pos, batch_neg], y=np.zeros((batch, output_dimension)))
                #hist = model.fit(x=[batch_anc, batch_pos, batch_neg], y=np.zeros((BATCH, output_dimension)))
                #print('hist', type(hist))
                #loss = 10

                batch_pos = []
                batch_neg = []
                batch_anc = []

                total_loss += loss
                n += 1
                if n % 10 == 0:
                    #print('n',n)
                    print("EPOCH: {} / {} n={} / {} LOSS for last ten n values: {}".format(epoch+1,epochs,n,num_random_triplets,total_loss))
                    total_loss = 0.0
                    #n = 0
    return model


'''
Train 
@parameters:
label_file: input CSV file
@wav_file (changed to folder with WAV files)
label_file_l2:
wav_file_l2:
out_folder="output"
perc_test=0.33
retrain=True
super_epochs=3
relabel=False
resample=10000
export_truth=True
'''
def train(X, y, label_dict, out_folder="output", perc_test=0.2, retrain=True, super_epochs=3, relabel=False, export_truth=False, continue_training=False, epochs=100, latent_dim = 128, batch=10, num_triplets = 120, num_classes = 8):
    print("######### General info #########")
    print('Fraction reserved for testing:',perc_test)
    
    reverse = dict([(v, k) for k, v in label_dict.items()]) #reverse dictionary    
    print('label_dict=',label_dict)

    #num_examples, D, T = X.shape
    num_examples, T, D = X.shape #we use transpose
    WINDOW_PARAM = (T, D, 1) #T is num_time_frames and D is num_freq_bins
    #WINDOW_PARAM = (D, T, 1) #T is num_time_frames and D is num_freq_bins

    #by_label is a dic with the instances of each class
    by_label = {} #initialize dictionary
    for i in range(num_examples):
        #this_label = reverse[y[i]]
        this_label = y[i]
        if this_label not in by_label:
            by_label[this_label] = [] #initialize list
        #print('------X[i]',X[i])
        by_label[this_label].append(X[i])
    print('by_label keys', by_label.keys())
    print('Histogram:')
    print([(k, len(v)) for k, v in by_label.items()])
    
    if retrain:    
        print("######### Training the 3 networks #########")
        #https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=perc_test, random_state=42, shuffle=True, stratify=y)
        #x_train = np.stack(x_train).reshape(len(x_train), T, D, 1)
        #x_test  = np.stack(x_test).reshape(len(x_test), T, D, 1)    
        #y_train = np.array(y_train)
        #y_test  = np.array(y_test)
        print('x_train',x_train.shape)
        print('x_test',x_test.shape)
        print('y_train',y_train)
        print('y_test',y_test)
        
        if continue_training: #read trained files
            base_encoder = load_model('{}/base_encoder.h5'.format(out_folder))
            enc = load_model('{}/encoder.h5'.format(out_folder))
        else: #initialize with random weights
            base_encoder = encoder(WINDOW_PARAM, latent_dim, CONV_PARAM)    
            enc = window_encoder(WINDOW_PARAM, base_encoder, latent_dim)

        print("base_encoder.summary")
        base_encoder.summary()        
        print("enc.summary")
        enc.summary()

        accuracy_supervised    = []
        accuracy_nn_supervised = []
        accuracy_siamese       = []
        accuracy_ae            = []
        #note that, while training, the encoder and base_encoder are not
        #initialized at each super_epoch, but the autoencoder and supervised classifier are
        for i in range(0, super_epochs):               
            print("######### Super epoch = ", (i+1) , "/", super_epochs, "#########")
            #by_label is a dic with the instances of each class         
            if True: #disable if do not want to train via triplets
                #num_random_triplets = 200 #total number of triplets to be created
                #num_random_triplets = 1000 * len(x_train) // BATCH #total number of triplets to be created
                #num_random_triplets = 10 * len(x_train) // BATCH #total number of triplets to be created
                num_random_triplets = num_triplets #total number of triplets to be created
                print("######### Improving encoder via triplet loss #########")
                siamese = train_triplets(enc, by_label, WINDOW_PARAM, num_random_triplets, epochs=epochs)
            else: #load already trained networks
                enc = load_model('{}/encoder.h5'.format(out_folder))
                base_encoder = load_model('{}/base_encoder.h5'.format(out_folder))
                #siamese = triplet_model(WINDOW_PARAM, enc, latent_dim) #create the model based on the given encoder
                #Loading a model with custom loss is tricky, I had to read:
                #https://stackoverflow.com/questions/60530304/loading-custom-model-with-tensorflow-2-1
                #https://stackoverflow.com/questions/60530304/loading-custom-model-with-tensorflow-2-1
                siamese = load_model('{}/siam.h5'.format(out_folder), custom_objects={'TripletLoss': TripletLoss},  compile=False)

            if not os.path.exists(out_folder):
                os.makedirs(out_folder)
                print("Created folder",out_folder)

            print("######### Saving current networks and filters based on triplet loss training #########")
            siamese.save('{}/siam.h5'.format(out_folder))
            enc.save('{}/encoder.h5'.format(out_folder))    
            base_encoder.save('{}/base_encoder.h5'.format(out_folder))            
            enc_filters(enc, N_FILTERS, N_BANKS, "{}/filters_siam.png".format(out_folder))                

            #need to learn what is doing below
            acc_siam = neighbours_encoder(enc, x_train, y_train, x_test, y_test, label_dict, "siamese", out_folder, BATCH=batch)
            accuracy_siamese.append(acc_siam)

            if False: #train auto encoder            
                print("######### Train autoencoder (and improve base_encoder and encoder) #########")
                ae          = auto_encoder(WINDOW_PARAM, enc, latent_dim, CONV_PARAM)    
                print("(auto_encoder) ae.summary")
                ae.summary()
                hist        = ae.fit(x=x_train, y=x_train, batch_size=BATCH, epochs=EPOCHS, shuffle=True)

                print("######### Saving current networks and filters based on auto encoder training #########")
                ae.save('{}/ae.h5'.format(out_folder))
                enc.save('{}/encoder.h5'.format(out_folder))        
                base_encoder.save('{}/base_encoder.h5'.format(out_folder))
                enc_filters(enc, N_FILTERS, N_BANKS, "{}/filters_ae.png".format(out_folder))                
                plot_tensorflow_hist(hist, "{}/history_train_ae.png".format(out_folder))

                print("######### Saving reconstructions based on auto encoder and predicted labels #########")
                visualize_dataset(ae.predict(x_test, batch_size=BATCH), "{}/reconstructions.png".format(out_folder), D, T)
                acc_ae = neighbours_encoder(enc, x_train, y_train, x_test, y_test, label_dict, "aute encoder", out_folder)
                accuracy_ae.append(acc_ae)
                #enc.save('{}/encoder.h5'.format(out_folder)) #already saved
                pkl.dump(label_dict, open('{}/labels.pkl'.format(out_folder), "wb"))

            print("######### Train classifier based on encoder output (and improve base_encoder and encoder) #########")
            model       = classifier(WINDOW_PARAM, enc, latent_dim, num_classes, CONV_PARAM) 
            print('model (classifier).summary()')
            model.summary()
            hist        = model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), batch_size=batch, epochs=epochs, shuffle=True)

            print("######### Saving current networks and filters based on classifier trained with supervised learning #########")
            model.save('{}/supervised.h5'.format(out_folder))
            enc.save('{}/encoder.h5'.format(out_folder))     
            base_encoder.save('{}/base_encoder.h5'.format(out_folder))
            enc_filters(enc, N_FILTERS, N_BANKS, "{}/filters_supervised.png".format(out_folder))        
            plot_tensorflow_hist(hist, "{}/history_train_supervised.png".format(out_folder))        
            acc_nn = neighbours_encoder(enc, x_train, y_train, x_test, y_test, label_dict, "classifier", out_folder)
            accuracy_nn_supervised.append(acc_nn)

            if relabel: #this is strange, but is disabled (False)
                prediction = model.predict(x_train)
                y_train    = prediction.argmax(axis=1)

            print("######### Saving confusion matrix based on classifier trained with supervised learning #########")                
            n = len(label_dict)        
            label_names = ["" for i in range(n)]
            for l, i in label_dict.items():
                label_names[i] = l
            prediction_test = model.predict(x_test)
            confusion = np.zeros((n,n))
            correct = 0
            for i in range(len(y_test)):
                pred = np.argmax(prediction_test[i])
                confusion[y_test[i], pred] += 1
            accuracy = np.sum(confusion * np.eye(n)) / len(y_test)
            plot_result_matrix(confusion, label_names, label_names, "confusion acc {}".format(accuracy))
            plt.savefig('{}/confusion_type.png'.format(out_folder))
            plt.close()
            accuracy_supervised.append(accuracy)

        plt.plot(accuracy_supervised, label="supervised")
        plt.plot(accuracy_nn_supervised, label="nn_supervised")
        plt.plot(accuracy_siamese, label="nn_siam")
        plt.plot(accuracy_ae, label="nn_ae")
        plt.legend()
        plt.title("Super Epochs")
        plt.savefig('{}/super_epoch_acc.png'.format(out_folder))
        plt.close()
    else:
        model = load_model('{}/supervised.h5'.format(out_folder))
        enc   = load_model('{}/encoder.h5'.format(out_folder))    

    by_label = dict([(k, enc.predict(np.stack(v), batch_size=10)) for k, v in by_label.items()])
    #clusters = dict([(k, cluster_model(v, out_folder, reverse[k], min_k = 8, max_k=None)) for k, v in by_label.items() if k != label_dict['NOISE']])
    clusters = dict([(k, cluster_model(v, out_folder, reverse[k], min_k = 8, max_k=None)) for k, v in by_label.items() if True])
    pkl.dump(clusters, open('{}/clusters_window.pkl'.format(out_folder),'wb'))
    print(f'Done Clustering: {[(k, v.cluster_centers_.shape) for k, v in clusters.items()]}')

    if False: #not working because we do not have the raw audios (object called ra)
        #create labels list based on y
        labels = []
        for i in range(num_examples):
            #this_label = reverse[y[i]]
            this_label = y[i]
            labels.append(this_label)

        #b = np.stack(instances)
        b = X
        h = enc.predict(b, batch_size=10)
        if export_truth:
            #x = labels
            x = y
        else:
            x = model.predict(b, batch_size=10)
        extracted = {}
        for n, i in enumerate(x):
            if n % 1000 == 0:
                print(f"{n} of {len(x)}")
            if export_truth:
                li = i
            else:
                li = int(np.argmax(i))
            l = reverse[li]
            if True: #l != 'NOISE':
                hn      = h[n].reshape(1, latent_dim)
                pred_hn = clusters[li].predict(hn)
                c = int(pred_hn[0])    
                if l not in extracted:
                    extracted[l] = {}
                if c not in extracted[l]:
                    extracted[l][c] = []
                if li != labels[n]:
                    l_true = reverse[labels[n]]
                    if l_true not in extracted[l]:
                        extracted[l][l_true] = []
                    extracted[l][l_true].append(ra[n])
                extracted[l][c].append(ra[n])

        for l, clusters in extracted.items():
            for c, audio in clusters.items():
                path = "{}/{}_{}.wav".format(out_folder, l, c)
                write(path, 44100, np.concatenate(audio))


if __name__ == '__main__':
    print("=====================================")
    print("Simplified WDP DS Pipeline")
    print("Original code by Daniel Kyu Hwa Kohlsdorf")

    parser = argparse.ArgumentParser()
    parser.add_argument('--general_dir',help='folder with wavs_labels.csv and labels_dictionary.json files',default='../general')
    parser.add_argument('--output_dir',help='output folder (default is the folder of the input file)')
    parser.add_argument('--super_epochs',type=int, help='number of super epochs', default=3)    
    parser.add_argument('--num_classes',type=int, help='number of classes (labels)', default=8)
    parser.add_argument('--epochs',type=int, help='number of epochs', default=100)    
    parser.add_argument('--batch',type=int, help='mini batch size', default=4)    
    parser.add_argument('--latent_dim',type=int, help='dimension of latent space (encoder output size)', default=128)        
    parser.add_argument('--num_triplets',type=int, help='number of random triplets', default=120)        
    parser.add_argument('--continue_training', action='store_true') #default is false    
    #required arguments    
    parser.add_argument('--input_file',help='input file with features',required=True)
    args = parser.parse_args()

    #print("python machine-learning/encoder_clusterer.py ../mel_D100T60/features_melD100T60_cutD6_60.hdf5 ../general/labels_dictionary.json ../mel_D100T60_output")

    general_dir = args.general_dir
    #label_file = os.path.join(general_dir, "wavs_labels.csv")
    labels_dic_file = os.path.join(general_dir, "labels_dictionary.json")
    continue_training = args.continue_training
    super_epochs = args.super_epochs
    num_classes = args.num_classes
    epochs = args.epochs
    batch = args.batch
    num_triplets = args.num_triplets
    latent_dim = args.latent_dim
    if args.output_dir == None:
        #create a default name
        output_folder = os.path.dirname(instances_file)
    else:
        output_folder = args.output_dir
    instances_file = args.input_file

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

    a_file = open(labels_dic_file, "r")
    print(a_file)
    label_dict_str = a_file.read()
    #https://appdividend.com/2022/01/29/how-to-convert-python-string-to-dictionary/
    label_dict = json.loads(label_dict_str)
    print("labels_dict", label_dict)
    print("labels_dict", type(label_dict))

    test_fraction = 0.2
    X_train, X_test, y_train, y_test, train_indices, test_indices = split_into_train_test_mixed_speakers(X, y, test_fraction)
    file_with_test_indices = os.path.join(output_folder,"test_indices.csv")
    np.savetxt(file_with_test_indices, test_indices, delimiter=",")
    file_with_train_indices = os.path.join(output_folder,"train_indices.csv")
    np.savetxt(file_with_train_indices, train_indices, delimiter=",")

    #train only with train set, and reserve test set for later
    train(X_train, y_train, label_dict, out_folder=output_folder,
        continue_training=continue_training, super_epochs=super_epochs,
        epochs=epochs, latent_dim = latent_dim, batch=batch,
        num_triplets=num_triplets, num_classes=num_classes)
