'''
This version assumes fixed dimension arrays obtained with
resize. Both frequency and time dimensions are fixed and called D and T, respectively.

From paper "An Auto Encoder For Audio Dolphin Communication" by
Daniel Kohlsdorf, dkohlsdorf@gmail.com
Denise Herzing, dlherzing@wilddolphinproject.org 
and Thad Starner College of Computing, Georgia Institute of Technology
Part of the "Wild Dolphin Project"
'''
import json
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse

from sklearn.model_selection import train_test_split

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
from data_generation import *
#from ml_util import read_instances_from_file

from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import *

VERBOSE = 1 #use 1 or 0

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
def ak_visualize_dataset(instances, output, D, T):
    plt.figure(figsize=(20, 20))
    for i in range(0, 100):
        xi = np.random.randint(0, len(instances))
        plt.subplot(10, 10, i + 1)
        plt.imshow(1.0 - instances[xi].reshape((D, T)).T, cmap='gray')
        #plt.imshow(1.0 - instances[xi].reshape((36, 130)).T, cmap='gray')
    plt.savefig(output)
    plt.clf()

'''
Write instances as vectors in latent space.
'''
#def write_latent_vectors(encoder, X, y, label_dict, name, out_folder):
#    X_latent = encoder.predict(X)

#def triplets(by_label, n = 50000):
def triplets(by_label, n = 5):
    '''
    Returns a list with n triplets (anchor, positive, negative) examples.
    Discussion by Daniel about triplet loss for siamese neural network architectures:
    http://daniel-at-world.blogspot.com/2019/07/implementing-triplet-loss-function-in.html
    '''
    l = list(by_label.keys()) #number of classes, e.g. 8
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
    (T, D, temp) = WINDOW_PARAM #T is num_time_frames and D is num_freq_bins
    print("Siamese network model")
    print(model.summary())
    for epoch in range(epochs):
        batch_pos = []
        batch_neg = []
        batch_anc = []
        n = 0
        total_loss = 0.0
        for anc, pos, neg in triplets(by_label,n=num_random_triplets):
            #print('anc, pos, neg',anc, pos, neg)
            batch_pos.append(get_segment_of_given_duration(pos, T))
            batch_neg.append(get_segment_of_given_duration(neg, T))
            batch_anc.append(get_segment_of_given_duration(anc, T))
            if len(batch_pos) == batch:
                batch_anc = np.stack(batch_anc)
                batch_pos = np.stack(batch_pos)
                batch_neg = np.stack(batch_neg)
                #note that all output labels are zero
                #https://stackoverflow.com/questions/62629997/difference-between-tensorflow-model-fit-and-train-on-batch
                #loss = model.train_on_batch(x=[batch_anc, batch_pos, batch_neg], y=np.zeros((BATCH,  256)))
                #z=[batch_anc, batch_pos, batch_neg]
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
Train the whole process.
'''
def train(X, y, label_dict, out_folder="output", perc_test=0.2, 
        super_epochs=3, continue_training=False,
        epochs=100, latent_dim = 128, batch=10, num_triplets = 120,
        triplet_epochs = 140):
    print("######### General info #########")
    print('Fraction reserved for testing:',perc_test)
    
    reverse = dict([(v, k) for k, v in label_dict.items()]) #reverse dictionary    
    print('label_dict=',label_dict)

    #num_examples, T, D = X.shape #we use transpose
    if type(X) == list: #X can be a list or a numpy array
        #num_examples = len(X)
        T = int(1e10)
        for i in range(num_examples):
            thisT, D = X[i].shape
            if thisT < T:
                T = thisT
    else:
        num_examples, T, D = X.shape #we use transpose

    WINDOW_PARAM = (T, D, 1) #T is num_time_frames and D is num_freq_bins

    #by_label is a dic with the instances of each class
    by_label = {} #initialize dictionary
    for i in range(num_examples):
        this_label = y[i]
        if this_label not in by_label:
            by_label[this_label] = [] #initialize list
        by_label[this_label].append(X[i])
    print('by_label keys', by_label.keys())
    print('Histogram:')
    print([(k, len(v)) for k, v in by_label.items()])
    
    if True:   
        print("######### Training the 3 networks #########")
        
        if continue_training: #read trained files
            base_encoder = load_model('{}/base_encoder.h5'.format(out_folder))
            enc = load_model('{}/encoder.h5'.format(out_folder))
        else: #initialize with random weights
            base_encoder = encoder(WINDOW_PARAM, latent_dim, CONV_PARAM)    
            enc = window_encoder(WINDOW_PARAM, base_encoder, latent_dim)
        
        # and define training / validation sets
        #https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
        #X and y below often correspond to the training set only
        x_train, x_validation, y_train, y_validation = train_test_split(X, y, 
                test_size=perc_test, shuffle=True, stratify=y) #random_state=42

        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
            print("Created folder",out_folder)

        print("Model base_encoder.summary")
        base_encoder.summary()        
        print("Model enc.summary")
        enc.summary()
        ae          = auto_encoder(WINDOW_PARAM, enc, latent_dim, CONV_PARAM)    
        print("(auto_encoder) ae.summary")
        ae.summary()

        accuracy_supervised    = []
        accuracy_nn_supervised = []
        accuracy_siamese       = []
        accuracy_ae            = []
        #note that, while training, the encoder and base_encoder are not
        #initialized at each super_epoch, but the autoencoder and supervised
        #classifier are
        for i in range(0, super_epochs):               
            print("######### Super epoch = ", (i+1) , "/", super_epochs, "#########")
            #by_label is a dic with the instances of each class         
            if True: #disable if do not want to train via triplets
                num_random_triplets = num_triplets #total number of triplets to be created
                print("######### Improving encoder via triplet loss #########")
                siamese = train_triplets(enc, by_label, WINDOW_PARAM, num_random_triplets, epochs=triplet_epochs, latent_dim = latent_dim)
            else: #load already trained networks
                enc = load_model('{}/encoder.h5'.format(out_folder))
                base_encoder = load_model('{}/base_encoder.h5'.format(out_folder))
                #siamese = triplet_model(WINDOW_PARAM, enc, latent_dim) #create the model based on the given encoder
                #Loading a model with custom loss is tricky, I had to read:
                #https://stackoverflow.com/questions/60530304/loading-custom-model-with-tensorflow-2-1
                #https://stackoverflow.com/questions/60530304/loading-custom-model-with-tensorflow-2-1
                siamese = load_model('{}/siam.h5'.format(out_folder), custom_objects={'TripletLoss': TripletLoss},  compile=False)

            print("######### Saving current networks and filters based on triplet loss training #########")
            #siamese.save('{}/siam.h5'.format(out_folder))
            enc.save('{}/encoder.h5'.format(out_folder))    
            base_encoder.save('{}/base_encoder.h5'.format(out_folder))            
            enc_filters(enc, N_FILTERS, N_BANKS, "{}/filters_siam.png".format(out_folder))

            if True:
                print("######### Train classifier based on encoder output (and improve base_encoder and encoder) #########")
                model       = classifier(WINDOW_PARAM, enc, latent_dim, num_classes, CONV_PARAM) 
                print('Model (classifier).summary()')
                model.summary()
                training_generator = DataGenerator(x_train, y_train, T, batch_size=batch, n_channels=1, n_classes=8, shuffle=True)
                validation_generator = DataGenerator(x_validation, y_validation, T, batch_size=batch, n_channels=1, n_classes=8, shuffle=True)

                model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join(out_folder, "checkpoint_model.hdf5"),
                    save_weights_only=False,
                    monitor='val_accuracy',
                    mode='max',
                    save_best_only=True)

                hist = model.fit(x=training_generator,
                        validation_data=validation_generator,
                        batch_size=batch,
                        verbose = VERBOSE,
                        callbacks=[model_checkpoint_callback],
                        epochs=epochs)

                print("######### Saving current networks and filters based on classifier trained with supervised learning #########")
                model.save('{}/supervised.h5'.format(out_folder))
                enc_filters(enc, N_FILTERS, N_BANKS, "{}/filters_supervised.png".format(out_folder))        
                plot_tensorflow_hist(hist, "{}/history_train_supervised.png".format(out_folder))

                print("######### Saving confusion matrix based on classifier trained with supervised learning #########")                
                n = len(label_dict)
                x_test_fixed_dim = convert_variable_into_fixed_duration(x_validation, T)
                label_names = ["" for i in range(n)]
                for l, i in label_dict.items():
                    label_names[i] = l
                prediction_test = model.predict(x_test_fixed_dim)
                confusion = np.zeros((n,n))
                for i in range(len(y_validation)):
                    pred = np.argmax(prediction_test[i])
                    confusion[y_validation[i], pred] += 1
                accuracy = np.sum(confusion * np.eye(n)) / len(y_validation)
                plot_result_matrix(confusion, label_names, label_names, "confusion acc {}".format(accuracy))
                plt.savefig('{}/confusion_type.png'.format(out_folder))
                plt.close()
                accuracy_supervised.append(accuracy)

            if True: #train auto encoder            
                print("######### Train autoencoder (and improve base_encoder and encoder) #########")
                hist        = ae.fit(x=X, y=X, batch_size=batch, epochs=epochs, shuffle=True)

            enc.save('{}/encoder.h5'.format(out_folder))     
            base_encoder.save('{}/base_encoder.h5'.format(out_folder))

        plt.plot(accuracy_supervised, label="supervised")
        plt.plot(accuracy_nn_supervised, label="nn_supervised")
        plt.plot(accuracy_siamese, label="nn_siam")
        plt.plot(accuracy_ae, label="nn_ae")
        plt.legend()
        plt.title("Super Epochs")
        plt.savefig('{}/super_epoch_acc.png'.format(out_folder))
        plt.close()


if __name__ == '__main__':
    print("=====================================")
    print("Simplified WDP DS Pipeline")
    print("Original code by Daniel Kyu Hwa Kohlsdorf")

    parser = argparse.ArgumentParser()
    parser.add_argument('--general_dir',help='folder with wavs_labels.csv and labels_dictionary.json files',default='../general')
    parser.add_argument('--super_epochs',type=int, help='number of super epochs', default=5)    
    parser.add_argument('--num_classes',type=int, help='number of classes (labels)', default=8)
    parser.add_argument('--epochs',type=int, help='number of epochs (does not include triplets training)', default=1000)
    parser.add_argument('--triplet_epochs',type=int, help='number of epochs for training triplets', default=1000)    
    parser.add_argument('--batch',type=int, help='mini batch size', default=3)    
    parser.add_argument('--latent_dim',type=int, help='dimension of latent space (encoder output size)', default=128)        
    parser.add_argument('--num_triplets',type=int, help='number of random triplets', default=800)        
    parser.add_argument('--continue_training', action='store_true') #default is false    
    parser.add_argument('--output_dir',help='output folder')
    #required arguments    
    parser.add_argument('--input_file',help='input file with features',required=True)
    args = parser.parse_args()

    #print("python machine-learning/encoder_clusterer.py ../mel_D100T60/features_melD100T60_cutD6_60.hdf5 ../general/labels_dictionary.json ../mel_D100T60_output")

    general_dir = args.general_dir
    label_file = os.path.join(general_dir, "wavs_labels.csv")
    labels_dic_file = os.path.join(general_dir, "labels_dictionary.json")
    continue_training = args.continue_training
    super_epochs = args.super_epochs
    num_classes = args.num_classes
    epochs = args.epochs
    batch = args.batch
    num_triplets = args.num_triplets
    triplet_epochs = args.triplet_epochs
    latent_dim = args.latent_dim
    instances_file = args.input_file
    input_folder = os.path.splitext(instances_file)[0]
    if args.output_dir == None:
        #create a default name
        output_folder = os.path.join(input_folder, "encoder_models")
    else:
        output_folder = args.output_dir

    if not os.path.exists(output_folder):
        os.makedirs(output_folder,exist_ok=True) #create folder if it does not exist
        print("Created output folder",output_folder)

    command_line_file = os.path.join(output_folder,'fixed_enc_commandline_args.txt')
    with open(command_line_file, 'w') as f:
        f.write(' '.join(sys.argv[0:]))
        #f.write(str(sys.argv[0:]))
    print("Saved file", command_line_file)

    #read the labels_file
    df_input_data = pd.read_csv(label_file) #read CSV label file using Pandas
    num_examples = len(df_input_data)
    if num_examples != 127:
        print("WARNING!!!! num_examples=", num_examples)

    #read dictionary
    a_file = open(labels_dic_file, "r")
    print(a_file)
    label_dict_str = a_file.read()
    #https://appdividend.com/2022/01/29/how-to-convert-python-string-to-dictionary/
    label_dict = json.loads(label_dict_str)
    print("labels_dict", label_dict)
    print("labels_dict", type(label_dict))

    #instances_file = r'justsounds_todoscorrigidos_normalizados\spectrogramD100T30.hdf5'
    #labels_dic_file = r'justsounds_todoscorrigidos_normalizados\labels_dictionary.json'
    #X, y = read_variable_duration_instances_from_files(input_folder,df_input_data,label_dict)
    X, y = read_instances_from_file(instances_file)
    y = y.astype(int) #convert output labels to integers
    num_examples = len(X)
    num_examples, T, D = X.shape #we use the transpose
    #print(num_examples, D, T)

    if continue_training: #read trained files
            #read previously used training / test sets to avoid leaking train examples into test
            file_with_test_indices = os.path.join(output_folder,"test_indices.csv")
            test_indices = np.genfromtxt(file_with_test_indices, delimiter=',')
            test_indices = test_indices.astype(int)
            file_with_train_indices = os.path.join(output_folder,"train_indices.csv")
            train_indices = np.genfromtxt(file_with_train_indices, delimiter=',')                
            train_indices = train_indices.astype(int)
            print("train_indices", train_indices, type(train_indices))
            print("test_indices", test_indices, type(test_indices))
            #print("X", X[0:3], type(X))
            y_train = y[train_indices]
            y_test = y[test_indices]
            X_test = [ X[i] for i in test_indices ]
            X_train = [ X[i] for i in train_indices ]
    else:
            test_fraction = 0.2
            X_train, X_test, y_train, y_test, train_indices, test_indices = split_into_train_test_mixed_speakers(X, y, test_fraction)
            file_with_test_indices = os.path.join(output_folder,"test_indices.csv")
            np.savetxt(file_with_test_indices, test_indices, delimiter=",")
            print("Wrote", os.path.join(output_folder,"test_indices.csv"))
            file_with_train_indices = os.path.join(output_folder,"train_indices.csv")
            np.savetxt(file_with_train_indices, train_indices, delimiter=",")

    #train only with train set, and reserve test set for later
    train(X_train, y_train, label_dict, out_folder=output_folder,
        continue_training=continue_training, super_epochs=super_epochs,
        epochs=epochs, latent_dim = latent_dim, batch=batch,
        num_triplets=num_triplets,
        triplet_epochs=triplet_epochs)
