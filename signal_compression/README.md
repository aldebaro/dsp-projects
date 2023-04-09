# Project guidelines

The guidelines can be found at the provided file.

## Signals Descriptions 

| Type | File Format | File name         | Unique Values | Original Sample Format | Max. Frequency (Hz)  | Duration (s) | Sampling Frequency (Fs) in Hz | Dynamic Range        | Formation Law                   | Lossless or Allowed SQNR(dB)    |
|------|-------------|-------------------|---------------|------------------------|----------------------|--------------|-------------------------------|----------------------|---------------------------------|---------------------------------|
| 1    | Binary      | original_1.double | 16            | double                 | 5 kHz                | 4            | 30k                           | [-7.49337, 7.4999]   | 2 sines                         | Lossless                        |
| 2    | Binary      | original_2.double | 32            | double                 | 400                  | 4            | 800                           | [5341.875, 5458.125] | Random Gaussian                 | Lossless                        |
| 3    | Binary      | original_3.double | 64            | double                 | 400                  | 16           | 3200                          | [-147.65, 147.656    | Upsampled Gaussian              | Lossless                        |
| 4    | Binary      | original_4.double | 57            | double                 | 6.67                 | 37.5         | 800                           | [-145.3, 145.3]      | Upsampled Gaussian              | Lossy, requires more than 13 dB |
| 5    | Binary      | original_5.double | 24            | double                 | 4 kHz                | 1.875        | 8000                          | [-126.25, 426.25]    | Mixture of Gaussians            | Lossless                        |
| 6    | ASCII       | original_6.txt | 12156         | text with 18 decimals                 | 4 kHz               | 1.875        | 8000                          | [-91.25, 15421.25]   | Mixture of Gaussians with trend | Lossless                        |

# Installation

Do not forget you need to install lasse-py.

## First you need to obtain a copy of lasse-py. Options:

 1) git clone it from https://github.com/lasseufpa/lasse-py

or

 2) download directly using github menu

## Then you need to install it with

1) python setup.py install

or

2) Configuring PYTHONPATH to point to its folder (see setpath.bat for the command set on Windows)
