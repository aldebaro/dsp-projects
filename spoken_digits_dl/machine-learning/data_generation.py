'''
Original code: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
'''
import numpy as np
import keras
import random

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, X, y, T, batch_size=32, n_channels=1, n_classes=8, shuffle=True):
        'Initialization'
        if type(X) == list:
            self.num_examples = len(X)
            thisT, self.D = X[0].shape
        else:
            self.num_examples, thisT, self.D = X.shape
        self.T = T
        self.batch_size = batch_size
        self.y = y
        self.X = X
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        #pre-allocate space to store a batch
        #self.X_batch = np.empty((self.batch_size, self.T, self.D, self.n_channels))
        self.X_batch = np.empty((self.batch_size, self.T, self.D))
        self.y_batch = np.empty((self.batch_size), dtype=int)

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.num_examples / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.num_examples)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_temp_indices):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Generate data
        # Store sample
        for i in range(len(list_temp_indices)):
            self.X_batch[i] = get_segment_of_given_duration(self.X[list_temp_indices[i]], self.T)
            
        # Store class
        self.y_batch = self.y[list_temp_indices]

        #return self.X_batch, keras.utils.to_categorical(self.y_batch, num_classes=self.n_classes)
        return (self.X_batch, self.y_batch)

def convert_variable_into_fixed_duration(X, T):
    num_examples = len(X)
    thisT, D = X[0].shape
    newX = np.zeros( (num_examples, T, D) )
    for i in range(num_examples):
        newX[i] = get_segment_of_given_duration(X[i], T)
    return newX

'''
Cut the spectrogram (of dimension T_variable x D) such that 
the output has a fixed dimension, imposed by T x D, where T <= T_variable. 
'''
def get_segment_of_given_duration(spectrogram, T):
    thisT, thisD = spectrogram.shape
    #print("This dimension:",thisT, thisD, T)
    if thisT < T:
        raise Exception("ERROR in logic: thisT < T:" + str(thisT) + "," + str(T))
    if thisT > T:
        random_start = random.randint(0, thisT - T)        
        random_end = random_start + T
        range = np.arange(random_start, random_end)
        #new_spectrogram = [ spectrogram[i,:] for i in range ]
        new_spectrogram = spectrogram[range,:]
        #print("AAA", range)
        return new_spectrogram
    else: #thisT == T and we do nothing
        return spectrogram

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--general_dir',help='folder with wavs_labels.csv and labels_dictionary.json files',default='../general')
    #required arguments    
    parser.add_argument('--input_folder',help='folder with feature input files',required=True)

    args = parser.parse_args()
    general_dir = args.general_dir
    label_file = os.path.join(general_dir, "wavs_labels.csv")
    labels_dic_file = os.path.join(general_dir, "labels_dictionary.json")
    input_folder = args.input_folder

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

    X, y = read_variable_duration_instances_from_files(input_folder,df_input_data,label_dict)

    # Generators
    T = 310
    training_generator = DataGenerator(X, y, T, batch_size=32, n_channels=1,
                 n_classes=10, shuffle=True)
    validation_generator = DataGenerator(partition['validation'], labels, **params)

    # Design model
    model = Sequential()
    [...] # Architecture
    model.compile()

    # Train model on dataset
    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=True,
                        workers=6)