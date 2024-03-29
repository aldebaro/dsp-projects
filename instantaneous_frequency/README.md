# Extraction of instantaneous frequencies using Synchrosqueezing in Python

The project main goal is to use the Synchrosqueezing method to analyse and extract instantaneous frequencies of signals that have variable frequencies over time.

# Basic information about the project

On the extraction of instantaneous frequencies from ridges in time-frequency representations of signals / Iatsenko, Dmytro, Peter VE McClintock, and Aneta Stefanovska. "On the extraction of instantaneous frequencies from ridges in time-frequency representations of signals." arXiv preprint arXiv:1310.7276 (2013).

Main datasets: 
- Sanctuary Soundscape Monitoring Project (SanctSound), url: https://sanctsound.ioos.us/sounds.html#dolphins-channel-islands
- Musics and respective Ground Truth generated with: https://github.com/weeping-angel/Mathematics-of-Music

Original source code: https://github.com/overlordgolddragon/ssqueezepy

Language: Python 3.10.4

Slide: https://docs.google.com/presentation/d/1qAXeufXGHlVJIEs1QKkt1ekBwBRU68T4HACOPWm_Fts/edit?usp=sharing

# Installation

Run `setup.py` to install the libraries and the test_signals dataset.

# Executing / performing basic analysis

- `ridge_generated_signals_ssq.py` - This script generates three cossenoid based signals, and it's respectives ground truth frequencies. Ridge curves are extracted from each signal then evaluated with the mean squared error metric.

Usage: `ridge_generated_signals_ssq.py --p penalty_for_frequency_change`

Default Parameters : `--p 100`

- `ridge_visualization_only.py` - This script plots ridge curves visualization of a loaded .wav signal.

Usage: `ridge_visualization_only.py --sf 'path_to_file.wav' --p penalty_for_frequency_change --nr number_of_ridges`

Default Parameters : `--sf './test_signals/visualization/dolphins/SanctSound_CI01_03_dolphins_20190904T064203Z.wav' --p 300000 --nr 3`

- `ridge_with_ground_truth.py` - This script plots and extracts a single ridge curve of a loaded .wav signal. The ridge extracted is evaluated with the original frequencies of the loaded signal (ground truth) using the mean squared error metric.

Usage: `ridge_with_ground_truth.py --sf 'path_to_file.wav' --ff 'path_to_ground_truth_file.npy' --p penalty_for_frequency_change`

Default Parameters : `--sf './test_signals/ground_truth/crescent_decrescent/music.wav' --ff './test_signals/ground_truth/crescent_decrescent/music_f.npy' --p 10`


# Credits

14/06/2022 - Wilson Antonio Cosmo Macedo - https://github.com/WCosmo

# References

- Modal Identification of Lightweight Pedestrian Bridges based on Time-Frequency Analysis. Jansen, Andreas. 10.13140/RG.2.2.17116.54407. (2016).

- On the extraction of instantaneous frequencies from ridges in time-frequency representations of signals / Iatsenko, Dmytro, Peter VE McClintock, and Aneta Stefanovska. "On the extraction of instantaneous frequencies from ridges in time-frequency representations of signals." arXiv preprint arXiv:1310.7276 (2013)

- A generative model and a generalized trust region Newton method for noise reduction. Computational Optimization and Applications. Pulkkinen, Seppo & Mäkelä, Marko & Karmitsa, Napsu. 57. 129-165. 10.1007/s10589-013-9581-4. (2014).

- Short Time Fourier Transform using Python and Numpy, 2014. Available in: <https://kevinsprojects.wordpress.com/2014/12/13/short-time-fourier-transform-using-python-and-numpy/>

- How to Play Music Using Mathematics in Python, 2020. Available in: <https://towardsdatascience.com/mathematics-of-music-in-python-b7d838c84f72/> 

