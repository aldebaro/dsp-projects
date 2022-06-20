# Digital Signal Processing for ECG Classification Using Wavelet Transform

Brief description of your project, indicating its goal

# Basic information about the project

Main paper / reference: Hannun, A. Y., Rajpurkar, P., Haghpanahi, M., Tison, G. H., Bourn, C., Turakhia, M. P., & Ng, A. Y. (2019). Cardiologist-level arrhythmia detection and classification in ambulatory electrocardiograms using a deep neural network. Nature medicine, 25(1), 65-69.

Main dataset: MIT-BIH Arrhythmia Database available at: https://physionet.org/content/mitdb/1.0.0/

Original code: https://github.com/physhik/ecg-mit-bih

Language: Python 3

# Dependency

* Python >= 3.8.10
* pip >= 22.1.2
* virtualenv==20.13.3
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

# Installation and Execution

To facilitate code execution, it is recommended to use a virtual environment for Python.

```
$ sudo pip3 install virtualenv
$ virtualenv venv_ecg
$ source venv_ecg/bin/activate
(venv_ecg) $ git clone https://github.com/aldebaro/dsp-projects.git
(venv_ecg) $ cd ecg_mitbih_classification/
(venv_ecg) $ pip install -r requrirements.txt
(venv_ecg) $ python src/data.py --downloading True
(venv_ecg) $ python src/train.py
```

# Credits

20/06/2022 Lucas Damasceno

# References

The main references you used
