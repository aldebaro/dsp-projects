import os
import wave
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
import math

wavFiles = []

for root, dirs, files in os.walk('../DataSet/'):
    for file in files:
        if (file.endswith('.wav')):
            wavFiles.append(os.path.join(root, file))

durationTimes = []

def getAudioData(wav_obj):
    n_samples = wav_obj.getnframes()
    sample_freq = wav_obj.getframerate()

    t_audio = n_samples/sample_freq

    signal_wave = wav_obj.readframes(n_samples)

    signal_array = np.frombuffer(signal_wave, dtype=np.int16)

    l_channel = signal_array[0::2]
    r_channel = signal_array[1::2]

    times = np.linspace(0, n_samples/sample_freq, num=n_samples)

    audioData = {'n_samples': n_samples, 'samples_freq': sample_freq, 'signal_array': signal_array, 'times': times, 't_audio': t_audio, 'r_channel': r_channel, 'l_channel': l_channel}
    return audioData

for wavFile in wavFiles:
    wav_obj = wave.open(wavFile)
    audioData = getAudioData(wav_obj)

    segment_size = 100

    # Não pega todo o audio por causa do math.floor
    segments_number = math.floor(len(audioData['signal_array'])/100)

    segments = np.empty([segments_number, math.floor(len(audioData['signal_array'])/100)])

    for i in range(segments_number):
        pointer = 0
        for j in range(segment_size*(i+1)-100, segment_size*(i+1)):
            segments[i, pointer] = audioData['signal_array'][j]
            pointer += 1

    energy = []
    for i in range(segments_number):
        energy.append((norm(segments[i])**2))

    plt.figure(i)
    plt.subplot(211)   
    plt.ylabel('Energy')
    plt.xlabel('Segments')
    plt.plot(energy)

    plt.title(f'Signal wave and energy segments of {wavFile}')
    plt.subplot(212)
    plt.ylabel('Amplitude')
    plt.xlabel('Samples')
    plt.plot(audioData['signal_array'])

    plt.show()


