import os
import wave
import numpy as np
import matplotlib.pyplot as plt
import argparse

wavFiles = []
amplitudes = []

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output')
parser.parse_args('--output 1'.split())

args = vars(parser.parse_args())

input_folder = 'DataSet'
output_folder = args['output']

if(os.path.exists('../' + output_folder)): 
    print('Output folder already exists!')
    exit()

for root, dirs, files in os.walk(f'../{input_folder}/'):
    for file in files:
        if (file.endswith('.wav')):
            wavFiles.append(os.path.join(root, file))

def getAudioData(wav_obj):
    n_samples = wav_obj.getnframes()
    sample_freq = wav_obj.getframerate()
    signal_wave = wav_obj.readframes(n_samples)
    signal_array = np.frombuffer(signal_wave, dtype=np.int16)

    audioData = {'n_samples': n_samples, 'samples_freq': sample_freq, 'signal_array': signal_array,}
    return audioData

for wavFile in wavFiles:
    wav_obj = wave.open(wavFile)
    audioData = getAudioData(wav_obj)

    min_aplitude = audioData['signal_array'].min()
    max_aplitude = audioData['signal_array'].max()
    mean_aplitude = audioData['signal_array'].mean()

    plt.ylabel('Amplitude value')
    plt.title(f'Amplitude histogram of {wavFile}')
    plt.hist(audioData['signal_array'])

    output_file = wavFile.replace(input_folder, output_folder)
    output_file = output_file.replace('.wav', '.png')
    output_file_folder = output_file.split('/')
    del output_file_folder[-1]
    output_file_folder = '/'.join(output_file_folder)

    #print(output_file_folder)

    if not (os.path.exists(output_file_folder)): os.makedirs(output_file_folder)

    #print(output_file)
    plt.savefig(output_file)
    plt.close()