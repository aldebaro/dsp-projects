# Digital Signal Processing for ECG Classification Using Wavelet Transform

Brief description of your project, indicating its goal

# Basic information about the project

Main paper / reference: Hannun, A. Y., Rajpurkar, P., Haghpanahi, M., Tison, G. H., Bourn, C., Turakhia, M. P., & Ng, A. Y. (2019). Cardiologist-level arrhythmia detection and classification in ambulatory electrocardiograms using a deep neural network. Nature medicine, 25(1), 65-69.

Main dataset: MIT-BIH Arrhythmia Database available at: https://physionet.org/content/mitdb/1.0.0/

Original code: https://github.com/physhik/ecg-mit-bih

Language: Python3

Slide: https://docs.google.com/presentation/d/1z3Sb4xVYEOk6fhB9vtsaUwbogzc4K8G5f5dTTfcNSDQ/edit?usp=sharing

# Dependency

## System

* Python >= 3.9.0
* pip >= 21.2.4
* conda >= 4.12.0

## Libraries
* keras==2.9.0
* tensorflow==2.9.1
* scikit-learn==1.1.1
* wfdb==3.4.1
* deepdish==0.3.7
* scipy==1.8.1
* numpy==1.22.4
* tqdm==4.64.0
* six==1.16.0
* Flask==2.1.2
* gevent==21.12.0
* werkzeug==2.1.2
* wget==3.2
* joblib==1.1.0
* skorch==0.11.0
* tensorboard==2.9.1
* PyWavelets==1.3.0
* opencv-contrib-python==4.6.0.66
* torch==1.12.0

# Installation and Execution

To facilitate code execution, it is recommended to use a conda virtual environment for Python.

Create and activate a conda environment with the following commands
```
(base) $ conda create -n ecg python==3.9
(base) $ conda activate ecg
```

Inside the conda environment created, install PyTorch with the appropriated command below

If you have a dedicated GPU, use the command:
```
(ecg) $ conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

If you don't have a dedicated GPU, use the command:
```
(ecg) $ conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

Clone the repository and enter on the ecg_mitbih_classification folder
```
(ecg) $ git clone https://github.com/aldebaro/dsp-projects.git
(ecg) $ cd ecg_mitbih_classification/
```

Install required libraries with the following command
```
(ecg) $ pip install -r requirements.txt
```

To generate results without Wavelet Transform use the following commands
```
(ecg) $ cd not_using_wavelet_transform/
(ecg) $ python data.py --downloading True
(ecg) $ python train.py
```

To generate results with Wavelet Transform use the following commands
```
(ecg) $ cd using_wavelet_transform/
(ecg) $ python preprocessing.py
(ecg) $ python trained.py
```

# Credits

20/06/2022 Lucas Damasceno

# References

The main references you used
