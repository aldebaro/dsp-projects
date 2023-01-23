'''
Create label file for the Free Spoken Digit Dataset (FSDD) dataset.
See https://github.com/Jakobovski/free-spoken-digit-dataset
'''
# The goal here is to create a CSV file with a header that can be
# interpreted by the code in this package.
# 
# Example:
# python automation/fsdd_dataset_create_label_file.py ../wav ../general
#
# where wav is the folder with input files, and general is the output folder.

import glob
import os.path
import numpy as np
import librosa
import sys
import pandas as pd
import json

show_histograms = True #use True to list histograms in the end

def fsdd_write_label_dictionary(label_file, output_labels_dic_file):
    #read input file
    df = pd.read_csv(label_file) #read CSV label file using Pandas
    label_dict = {} #dictionary
    for i in range(10): #go over all digits
        label_dict[i] = i
    a_file = open(output_labels_dic_file, "w")
    json.dump(label_dict, a_file)
    a_file.close()

'''
Parse file name. For example:
9_yweweler_6.wav
returns
9, yweweler, 6
'''
def parse_file_name(file_name):
    tokens = file_name.split("_")
    if len(tokens) != 3:
        raise Exception("Could not parse " + file_name)
    return tokens

'''
Find one substring from given list of substrings in a string
'''
def OLD_find_substring_in_string(string, substrings_list):
    num_substrings = len(substrings_list)
    found = False
    this_substring = None
    this_substring_index = -1
    for i in range(num_substrings):
        substring = substrings_list[i].lower()
        if string.find(substring) != -1:
                if found == True:
                    print("ERROR: found two substrings", substring, "and", this_substring, "in string", string)
                    exit(-1)
                found = True
                this_substring = substrings_list[i]
                this_substring_index = i
    if not found:
            print("ERROR: could not find any label in", string)
            exit(-1)
    return this_substring, this_substring_index

#input folder with wav files (the extension must be wav, not WAV or something else)
def write_to_file(folder, fp):
    expected_Fs = 8000 #expected sample rate (all files should have it)
    labels = ["0","1","2","3","4","5","6","7","8","9"] #,"NOISE"]
    speakers = set() #set of speakers. Empty for now
    header = "filename,annotation,offset,starts,stops,speaker" #assumed header for labels

    histogram = np.zeros((len(labels),),dtype='int')

    #print("Reading folder", folder)
    #print(header)
    fp.write(header)
    fp.write('\n')
    iterator = glob.glob(os.path.join(folder,"*.wav"))
    num_wav_files = len(iterator)
    print("Found", num_wav_files, "files with extension wav in folder", folder)
    for wave_file in iterator:
        #print("Processing {}...".format(wave_file))

        #read waveform and check sampling frequency
        x, Fs = librosa.load(wave_file, sr=None) #x is a numpy array
        if Fs != expected_Fs:
            print("ERROR:", wave_file, "has sampling frequency",Fs)
            exit(-1)

        #process the file name, searching for its label and speaker in filename
        filename = os.path.basename(wave_file)
        filename_without_extension = os.path.splitext(filename)[0]
        filename_without_extension = filename_without_extension.lower()

        this_label, this_speaker, this_counter = parse_file_name(filename_without_extension)
        #print(this_label, this_speaker, this_counter)

        this_label_index = int(this_label)
        histogram[this_label_index] += 1

        speakers.update(this_speaker)
        
        #It is assumed:    
        #header = "filename,annotation,offset,starts,stops" #assumed header for labels
        #print(filename,",",this_label,",",this_label_index,",0,0,",x.shape[0],",",this_speaker,sep='')
        output_string = filename + "," + this_label + ",0,0," + str(x.shape[0]) + "," + this_speaker
        fp.write(output_string)
        fp.write('\n')

    if show_histograms: #enable to see the histogram
        print("Label Histogram:")
        print(labels)
        print(histogram)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("ERROR!")
        print("Usage: input_wav_folder output_folder")
        print("Example:")
        print(r"python create_label_file.py ../wav ../output/")
        exit(1)
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

    if not os.path.exists(input_folder):
        print("ERROR: folder", input_folder,"does not exist!")
        exit(-1)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder,exist_ok=True) #create folder if it does not exist
        print("Created output folder",output_folder)

    output_file = os.path.join(output_folder, "wavs_labels.csv")
    with open(output_file, 'w') as fp:
        write_to_file(input_folder, fp)
    fp.close()
    print("Wrote file", output_file)

    output_labels_dic_file = os.path.join(output_folder, "labels_dictionary.json")
    label_file = output_file
    fsdd_write_label_dictionary(label_file, output_labels_dic_file)
    print("Wrote file", output_labels_dic_file)
