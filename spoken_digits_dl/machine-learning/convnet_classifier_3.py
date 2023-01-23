'''Trains a deep NN for whistled speech classification.

Assumes this file structure:
https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory
and this Keras pipeline:
https://keras.io/examples/vision/image_classification_from_scratch/

See, for explanation about convnet and filters:
https://datascience.stackexchange.com/questions/16463/what-is-are-the-default-filters-used-by-keras-convolution2d
and
http://cs231n.github.io/convolutional-networks/
'''
from __future__ import print_function

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
#from keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import numpy as np
import tensorflow as tf
from tensorflow import keras

#one can disable the imports below if not plotting / saving
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

epochs = 100 #number of epochs (passing over training set)
numClasses = 8 #total number of labels
# from keras pipeline tutorial:
#should approximately match the actual aspect ratio of the images that is approximately=2
num_rows = 800
num_columns = 1500
image_size = (num_rows, num_columns)
batch_size = 1
dropout_probability = 0.01 #to combat overfitting

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "/main_folder/train",
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "/main_folder/test",
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)
print("val_ds",val_ds)
print("train_ds",train_ds)
print("val_ds.cardinality().numpy()=",val_ds.cardinality().numpy()) 
print("train_ds.cardinality().numpy()=",train_ds.cardinality().numpy()) 

if False:
    plt.figure(figsize=(10, 10))
    #for images, labels in train_ds.take(43):
    for images, labels in val_ds.take(8):
        for i in range(2):
            ax = plt.subplot(2, 1, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(int(labels[i]))
            plt.axis("off")
        plt.show()
        #print("Type <ENTER> for next")
        #input()

#Convert output labels to one-hot encoding according to
#https://stackoverflow.com/questions/65328084/one-hot-encode-labels-of-tf-data-dataset
def _map_func(text, label):
    numClasses = 8
    label = tf.one_hot(label, numClasses, name='label', axis=-1)
    return text, label

#run the mapping above over the whole sets
train_ds = train_ds.map(_map_func)
val_ds = val_ds.map(_map_func)

print("val_ds",val_ds)
print("train_ds",train_ds)
print("val_ds.cardinality().numpy()=",val_ds.cardinality().numpy()) 
print("train_ds.cardinality().numpy()=",train_ds.cardinality().numpy()) 

model = Sequential()

model.add(Conv2D(30, kernel_size=(20,20),
                 activation='relu',
				 strides=[3,3],
				 padding="SAME",
                 #consider we have color images with 3 channels (depth):
                 input_shape=(num_rows, num_columns,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))                 
model.add(Dropout(dropout_probability))
model.add(Conv2D(30, (10, 10), padding="SAME", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))                 
model.add(Dropout(dropout_probability))
model.add(Conv2D(30, (4, 4), padding="SAME", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))                 
model.add(Conv2D(30, (2, 2), padding="SAME", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))                 
model.add(Dropout(dropout_probability))
model.add(Flatten())
model.add(Dense(numClasses, activation='softmax')) #softmax for probability

model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
              
#install graphviz: sudo apt-get install graphviz and then pip install related packages
#plot_model(model, to_file='classification_model.png', show_shapes = True)

# compile model.
#model.compile(loss='mean_squared_error',
#              optimizer=Adagrad(),
#              metrics=['accuracy','mae'])

model.load_weights('/nethome/pinheirs/Documents/papi/easyspeech/mots8/spectrogram_copy/save_at_{epoch}.h5')


callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
]

#history = model.fit(X_train, y_train,
 #                   batch_size=batch_size,
  #                  epochs=epochs,
   #                 verbose=1,
    #                shuffle=True,
     #               validation_split=validationFraction)
                    #validation_data=(X_test, y_test))

history = model.fit(
    train_ds, 
    epochs=epochs, 
    callbacks=callbacks, 
    validation_freq = 1,
    #validation_data=val_ds,
    validation_data=train_ds
)

# print results
#score = model.evaluate(X_test, y_test, verbose=0)
#print(model.metrics_names)
#print('Test loss rmse:', np.sqrt(score[0]))
#print('Test accuracy:', score[1])
#print(score)

print("Contents in history:", history.history.keys())

#save accuracies for plotting in external programs, could also save the losses:
val_acc = history.history['val_accuracy']
acc = history.history['accuracy']
f = open('classification_output.txt','w')
f.write('validation_acc\n')
f.write(str(val_acc))
f.write('\ntrain_acc\n')
f.write(str(acc))
f.close()

#enable if want to plot images
if False:
    from keras.utils import plot_model
    import matplotlib.pyplot as plt

    pred_test = model.predict(X_test)
    for i in range(len(y_test)):
        if (original_y_test[i] != np.argmax(pred_test[i])):
            myImage = X_test[i].reshape(nrows,ncolumns)
            plt.imshow(myImage)
            plt.show()
            print("Type <ENTER> for next")
            input()
