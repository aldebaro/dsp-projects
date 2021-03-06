# Reconstruction of sparse time-frequency signals by preserving phase information

Analyze and reconstruct sparse time-frequency signals that are added to dense signals, such as certain noise, which will be reconstructed through the preservation of phase information that serves as a consensus for signal information stability.

# Basic information about the project

Main paper: T. J. Gardner and M. O. Magnasco, "Sparse time-frequency representations", Proc. Nat. Acad. Sci. USA, vol. 103, no. 16, pp. 6094-6099, 2006.

Main dataset: 

* Radek Osmulski: https://github.com/earthspecies/spectral_hyperresolution/tree/main/data
* Maggie Zhang: https://github.com/maggie-jiayizhang/hyper_resolution_spectrogram/tree/main/test_audio
* Ground Truth: https://github.com/weeping-angel/Mathematics-of-Music

Original code: 
* Radek Osmulski: https://github.com/earthspecies/spectral_hyperresolution
* Maggie Zhang: https://github.com/maggie-jiayizhang/hyper_resolution_spectrogram

Language: Jupyter notebook in Python and Matlab.

Link to slides: https://docs.google.com/presentation/d/1-CIlU9wCmOhtC4grihvUbIT2FLDIvwCmZCqzk-n8mV4/edit#slide=id.p

# Installation

It is necessary to install the following packages in order to run the code next:

`pip install librosa`

`pip3 install torch torchvision torchaudio`

`pip install git+https://github.com/earthspecies/spectral_hyperresolution.git@main`

# Executing / performing basic analysis

### Steps to run the code present at: https://github.com/earthspecies/spectral_hyperresolution

To run the main file (linear_reassignment_overview), it is necessary to make some changes in the code, which are listed below:

The last line command of cell 4 should be changed from "waveplot" to "waveshow".

Also, you need to run the next cells with the CPU instead of CUDA, so you have to change the last parameter of the last row from cell eight onwards to 'cpu'.

### Steps to run the code present at: https://github.com/maggie-jiayizhang/hyper_resolution_spectrogram

Open the directory of code and pass the comand:

`python3 example.py --data_dir your_dir`

Where the *your_dir* is the data folder.

# Credits

06/20/2022 - Marcos Davi Lima da Silva - https://github.com/marcossilva1309

# References

Sparse time-frequency representations by Timothy J. Gardner and Marcelo O. Magnasco / reference: T. J. Gardner and M. O. Magnasco, "Sparse time-frequency representations", Proc. Nat. Acad. Sci. USA, vol. 103, no. 16, pp. 6094-6099, 2006.

Patterson RD. A pulse ribbon model of monaural phase perception. J Acoust Soc Am. 1987 Nov;82(5):1560-86. doi: 10.1121/1.395146. PMID: 3693696.

Flandrin, Patrick, Francois Auger, and Eric Chassande-Mottin. "Time-frequency reassignment: from principles to algorithms." Applications in time-frequency signal processing. CRC Press, 2018. 179-204.
