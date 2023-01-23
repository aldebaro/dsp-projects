'''
First, creates a argument parser to interpret command line arguments, utilized to know the output folder to store the images generated (-o or --o <folder>).
If the output folder already exists, the program stop to prevent overwriting files.
Then, search recursively all files with .wav extension and sub-directories, then store the paths of these files.
After that the code process each file to read the signal waveform and saves the plot in a png file on the choosen output folder.

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
import argparse

wavFiles = []
amplitudes = []

# Creates argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output')
parser.parse_args('--output 1'.split())

# Gets the args from the parses
args = vars(parser.parse_args())

input_folder = 'DataSet'
output_folder = args['output']

# If the output folder already exists, the code stops
if(os.path.exists('../' + output_folder)): 
    print('Output folder already exists!')
    exit()

# Search reursively the files and store in a list
for root, dirs, files in os.walk(f'../{input_folder}/'):
    for file in files:
        if (file.endswith('.wav')):
            wavFiles.append(os.path.join(root, file))

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

    # Creates the plot of current file waveform
    plt.ylabel('Amplitude value')
    plt.title(f'Amplitude histogram of {wavFile}')
    plt.hist(audioData['signal_array'])

    # Sets the output folder, utilizing the same folder structure but in a different output "root" folder
    output_file = wavFile.replace(input_folder, output_folder)
    output_file = output_file.replace('.wav', '.png')
    output_file_folder = output_file.split('/')
    del output_file_folder[-1]
    output_file_folder = '/'.join(output_file_folder)

    # If the path doesn't exists, create
    if not (os.path.exists(output_file_folder)): os.makedirs(output_file_folder)

    # Save the plot in a png file to the output folder
    plt.savefig(output_file)
    plt.close()