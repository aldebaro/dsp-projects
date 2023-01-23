'''
Search recursively all files with .wav extension and sub-directories, then store the paths of these files.
After that the code process each file to read the signal waveform and then find the duration in seconds.
With this data, we describe some statistcs of each file, containing the duration and minimum, maximum and mean amplitude,
displaying the amplitude to each waveform and in the end we describe the minimum, maximum and mean duration of the files.

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

durationTimes = np.empty(len(wavFiles))

# Function to get the waveform data of each file
def getAudioData(wav_obj):
    n_samples = wav_obj.getnframes()
    sample_freq = wav_obj.getframerate()
    signal_wave = wav_obj.readframes(n_samples)
    signal_array = np.frombuffer(signal_wave, dtype=np.int16)

    audioData = {'n_samples': n_samples, 'samples_freq': sample_freq, 'signal_array': signal_array,}
    return audioData

for wavFile in wavFiles:
    # Here we could have utilized other options like scipy.wave or wavy
    wav_obj = wave.open(wavFile)
    audioData = getAudioData(wav_obj)

    # Find the duration of the audio file in seconds, storing in a list
    audioData['duration'] = audioData['n_samples'] / float(audioData['samples_freq'])
    duration = audioData['duration']

    np.append(durationTimes, audioData['duration'])

    # Describes min, max and mean amplitude of current file
    min_aplitude = audioData['signal_array'].min()
    max_aplitude = audioData['signal_array'].max()
    mean_aplitude = audioData['signal_array'].mean()

    print(f'\nFile: {wavFile}')
    print(f'Duration: {duration}s')
    print(f'Min amplitute: {min_aplitude}')
    print(f'Max amplitute: {max_aplitude}')
    print(f'Mean amplitute: {mean_aplitude}')

    # Plot a histogram of amplitude 
    plt.ylabel('Amplitude value')
    plt.title(f'Amplitude histogram of {wavFile}')
    plt.hist(audioData['signal_array'])
    plt.show()

# Describes min, max and mean duration for all files in dataset
print(f'\nMin duration: {min(durationTimes)}s')
print(f'Max duration: {max(durationTimes)}s')
print(f'Mean duration: {np.nanmean(durationTimes)}s')

