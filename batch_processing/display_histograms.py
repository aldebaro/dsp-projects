import os
import wave
import numpy as np
import matplotlib.pyplot as plt

wavFiles = []
amplitudes = []

for root, dirs, files in os.walk('../DataSet/'):
    for file in files:
        if (file.endswith('.wav')):
            wavFiles.append(os.path.join(root, file))

durationTimes = np.empty(len(wavFiles))

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

    audioData['duration'] = audioData['n_samples'] / float(audioData['samples_freq'])
    duration = audioData['duration']
    #durationTimes.append(audioData['duration'])
    np.append(durationTimes, audioData['duration'])

    min_aplitude = audioData['signal_array'].min()
    max_aplitude = audioData['signal_array'].max()
    mean_aplitude = audioData['signal_array'].mean()

    print(f'\nFile: {wavFile}')
    print(f'Duration: {duration}s')
    print(f'Min amplitute: {min_aplitude}')
    print(f'Max amplitute: {max_aplitude}')
    print(f'Mean amplitute: {mean_aplitude}')

    plt.ylabel('Amplitude value')
    plt.title(f'Amplitude histogram of {wavFile}')
    plt.hist(audioData['signal_array'])
    plt.show()

print(f'\nMin duration: {min(durationTimes)}s')
print(f'Max duration: {max(durationTimes)}s')
print(f'Mean duration: {np.nanmean(durationTimes)}s')

