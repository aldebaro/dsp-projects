# hyper_resolution_spectrogram

This repository hosts a working Python3 implementation of the reassignment algorithm for generating *hyper-resolution spectrogram* from audio signals, described in [Gardner & Magnasco, 2006](./gardner_magnasco_2006.pdf). The resultant histograms are sprase time-frequency representations of the original signal, with resolution much higher than that of conventional histograms (usually generated with Short Time Fourrier Transform). This algorithm can be applied to analyzing a wide range of signals, ranging from click-like to tonal--it has been used in projects that examine dophin vocal communication and automatic disease detection through voice recordings, for examples.

There are several implementations within the *reassignment* folder, and *reassignment_linear.py* contains the implemenation closest to the original [Matlab code](./reassignmentgw.m), provided by Dr. Magnasco. For more detailed information, please read the specifications within each file.

An example routine is provided in the *example.py* file. You can run
```bash
$ python3 example.py --data_dir your_dir
```
for batch-processing of all audio signals in the directory of your choice. The routine also saves the histogram matrix (in .npy format) and an image representation for each audio (ending in .wav, but can be easily changed to any other extension). 

## Aside: 
The current implementation is based off of the original one and the Matlab code. The previous [implementation](./https://github.com/earthspecies/spectral_hyperresolution.git) by Radek Osmulski is deprecated due to an update in the PyTorch package. The author of this repository 

1. adapted the original implementation to the newer PyTorch methods
2. simplified the code (removed some unnecessary tensor allocations)
3. changed the functions to take in mono-channeled signals (instead of 2-channeled, in the original implementation)[^1]

All this work was completed under the supervision of Dr. Magnasco (co-author of the method) and support from Mr. Osmulski in summer 2021. The author would like to thank Mr. Osmulski and Dr. Magnasco for their code and patient explanations.

[^1]: This is to make the method more generic. To process multi-channeled signals, simply create a loop and call the method to analyze each channel separately.