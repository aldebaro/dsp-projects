# Spoken digits processing with deep learning

DSP and machine learning for studying deep learning applied to isolated word (digits from 0 to 9) recognition.

# Basic information about the project

Forked from https://github.com/sklautau/whistled-speech-analysis

Auxiliary software: Praat (https://www.fon.hum.uva.nl/praat/) and dataset https://www.kaggle.com/datasets/lazyrac00n/speech-activity-detection-datasets?resource=download

Language: Python 3

# Default dataset

We will assume the dataset FSSD (https://github.com/Jakobovski/free-spoken-digit-dataset).

Because the FSSD is relatively large, you can use just a small subset from its files by downloading small_fssd.zip (1.4 MB) from https://nextcloud.lasseufpa.org/s/Liq6RoikgjqFKC4

# Dependency

## System

* Python >= 3.9.0
TO-DO

## Libraries
* numpy==1.22.4

## Getting started

After cloning the project to a folder that we will call "root" here, create a folder "wav" in "root", with the waveform files. For that, after obtaining the full FSSD dataset or the subset from small_fssd.zip, put the .wav files in folder root\wav. If using the full FSSD, you can simply copy or move the contents of its folder called "recordings" to root\wav.

Then, from "root" folder, execute:

python execute_all.py

# Credits

# References

The main references you used
