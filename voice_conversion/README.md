# Voice Conversion Using Speech-to-Speech Neuro-Style Transfer

The paper goal its present a new voice conversion method based on a neural style transfer model of the mel-spectrogram.

# Basic information about the project

Main paper / reference: Ehab A. AlBadawy, Siwei Lyu, <"Voice Conversion Using Speech-to-Speech Neuro-Style Transfer">, University at Albany, SUNY, USA

Main dataset: [(https://groups.csail.mit.edu/sls/downloads/flickraudio/downloads.cgi)]

Original code: https://github.com/ebadawy/voice_conversion

slide: https://docs.google.com/presentation/d/1pgDC-qwCtOGZSIG5tjIRUbSUCfo6EeyqRD-j7djDP5E/edit#slide=id.g13331885fbe_0_5

Language: Python 

# Installation

pip install librosa

pip install tqdm

pip install webrtcvad

pip install nnmnkwii

# Executing / performing basic analysis

First of all, a paper was selected from github which it has the codes in python. Downloading files were saved in .zip and uploaded on the server generated on the google cloud platform, where the debian distribution was being simulated.

Use the `unzip` command to unzip the `voice_conversion-master.zip` and `wavenet_vocoder-master.zip`

command: `unzip [name].zip`

After unzipping the folders with the codes, then the dataset `flickr_audio.zip` was uploaded to the server (this might take few hours)

The enviroment must be structured as follow: `flickr_audio` `voice_conversion-master` `wavenet_vocoder-master`

Enviroment ready, the next step is use the preprocess/train/inference codes present in voice_conversion-master:
- Preprocess: `python preprocess.py --dataset [path/to/dataset] --test-size [float] --eval-size [float]`

- Training: `python train.py --model_name [name]  --dataset [path/to/dataset] --n_cpu 4`

- Inference: `python inference.py --model_name [name of the model] --epoch [epoch number] --trg_id [1 or 2] --wav [path/to/source_audio.wav]` or `--wavdir path/to/audio_directory`

# Possible solutions if there are problems installing nnmnkwii library

1st case:    `pip install nnmnkwii` or `python3 -m pip install nnmnkwii`

2nd case:    `sudo apt install git-all` (DEBIAN)

             `pip install git+https://github.com/r9y9/nnmnkwii` 
             
3rd Case:    'pip -v --no-cache-dir install nnmnkwii`


4th Case:    'pip install numpy / python3 -m pip install numpy'
             'pip install cython / python3 -m pip install cython'

5th Case:    `pip install nnmnkwii==0.1.0`

Feedback: none of this possible solution worked in the present work. This library was supposed to be used in separate wavenet training.

# Credits

14/06/2022 - Tatiane Ferraz Balbinot - your github URL

# References

Main reference: https://github.com/ebadawy/voice_conversion

Auxiliary reference: https://github.com/RussellSB/tt-vae-gan

WaveNet VoCoder source: https://github.com/r9y9/wavenet_vocoder 

Dataset: https://groups.csail.mit.edu/sls/downloads/flickraudio/downloads.cgi
