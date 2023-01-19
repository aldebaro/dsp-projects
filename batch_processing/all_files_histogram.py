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

    signal_wave = wav_obj.readframes(n_samples)

    signal_array = np.frombuffer(signal_wave, dtype=np.int16)

    audioData = {'n_samples': n_samples, 'signal_array': signal_array}
    return audioData

for wavFile in wavFiles:
    wav_obj = wave.open(wavFile)
    audioData = getAudioData(wav_obj)   
    plt.hist(audioData['signal_array'])

plt.title(f'Histogramas de {len(wavFiles)} ondas')
plt.ylabel('Recorrência')
plt.xlabel('Amplitude')
plt.show()