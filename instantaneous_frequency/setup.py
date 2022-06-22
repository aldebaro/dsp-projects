'''
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
reqs = subprocess.check_output ([sys.executable, '-m', 'pip', 'freeze'])
installed_packages = [r.decode().split('==')[0] for r in reqs.split()]
print(installed_packages)
print('\nLibrary install done\n----------------------------------------\n')

print('\nStarting data download\n----------------------------------------\n')
gdd.download_file_from_google_drive(file_id='10g5s9eQ2_Qap8zq-3or5NsGN_ZEbUZHn', dest_path='./signal.zip', unzip=True)
os.remove('./signal.zip')
print('\nData download done\n----------------------------------------\n')
