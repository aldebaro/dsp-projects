'''
Search recursively all files with .wav extension and sub-directories, then store the paths of these files.
After that the code process each file to compute a single histogram for the whole dataset.

The main module to read signals from the .wav files is the built in python wave, considering that we don't need to install anything new
this module was the best choice to begginers in python. As other options to extract the waveform from the files, we have the modules
scipy.wave or wavy (https://pypi.org/project/wavy/)

Credits:
- Dataset utilized to tests: https://www.kaggle.com/datasets/lazyrac00n/speech-activity-detection-datasets?resource=download

Authors:
- Giovanna Cunha
- Gabriel Moraes
- Valdinei Rodrigues
'''

import os
import wave
import numpy as np
import matplotlib.pyplot as plt

wavFiles = []
amplitudes = []

# Search reursively the files and store in a list
for root, dirs, files in os.walk('../DataSet/'):
    for file in files:
        if (file.endswith('.wav')):
            wavFiles.append(os.path.join(root, file))

# Function to get the waveform data of each file
def getAudioData(wav_obj):
    n_samples = wav_obj.getnframes()

    signal_wave = wav_obj.readframes(n_samples)

    signal_array = np.frombuffer(signal_wave, dtype=np.int16)

    audioData = {'signal_array': signal_array}
    return audioData

for wavFile in wavFiles:
    # Here we could have utilized other options like scipy.wave or wavy
    wav_obj = wave.open(wavFile)
    audioData = getAudioData(wav_obj)   
    plt.hist(audioData['signal_array'])

# Plots a single histogram of all wavefiles amplitudes in the dataset
plt.title(f'Histograms of {len(wavFiles)} waves')
plt.ylabel('Recurrence')
plt.xlabel('Amplitude')
plt.show()