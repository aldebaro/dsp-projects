'''
DSP - Final Project
Project Name: Extraction of instantaneous frequencies using Synchrosqueezing in Python
Student: Wilson Cosmo
Date: 22/06/2022

Script description: This script installs the libraries for the project and download a small dataset for testing.
'''

import os
import sys
import subprocess
from google_drive_downloader import GoogleDriveDownloader as gdd

print('\nStarting library install\n----------------------------------------\n')
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'numpy'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'matplotlib'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'scipy'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'ssqueezepy'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'googledrivedownloader'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'sklearn'])
reqs = subprocess.check_output ([sys.executable, '-m', 'pip', 'freeze'])
installed_packages = [r.decode().split('==')[0] for r in reqs.split()]
print(installed_packages)
print('\nLibrary install done\n----------------------------------------\n')
print('\nStarting data download\n----------------------------------------\n')
gdd.download_file_from_google_drive(file_id='15Z6ZK2TH3-0JvE0mcMBtsbmmsljafs1L', dest_path='./test_signals.zip', unzip=True)
os.remove('./test_signals.zip')
print('\nData download done\n----------------------------------------\n')
