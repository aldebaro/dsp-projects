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

Run setup.py to install the libraries and the test_signals dataset.

# Executing / performing basic analysis

- ridge_generated_signals_ssq.py - This script generates three cossenoid based signals, and it's respectives ground truth frequencies. Ridge curves are extracted from each signal then evaluated with the mean squared error metric.

Input Parameters: --p 'penalty for frequency change'

- ridge_visualization_only.py - This script plots ridge curves visualization of a loaded .wav signal.

Input Parameters: --sf 'path_to_file.wav' --p 'penalty for frequency change' --nr 'number of ridges'

- ridge_with_ground_truth.py - This script plots and extracts a single ridge curve of a loaded .wav signal. The ridge extracted is evaluated with the original frequencies of the loaded signal (ground truth) using the mean squared error metric.

Input Parameters: --sf 'path_to_file.wav' --ff 'path_to_ground_truth_file.npy' --p 'penalty for frequency change'


# Credits

14/06/2022 - Wilson Antonio Cosmo Macedo - https://github.com/WCosmo

# References

The main references you used
