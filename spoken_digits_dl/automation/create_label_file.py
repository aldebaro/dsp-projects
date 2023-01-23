'''
Find all .wav files in a directory and create CSV to become Pandas dataframe

It prepares the file for Daniel's project:
# wdp-ds - Data Science For The Wild Dolphin Project
# https://github.com/dkohlsdorf/wdp-ds/releases/tag/v1.0
'''
#
# The goal here is to create a CSV file with a header that can be
# interpreted by Daniel's code. Hence annotation is the class label (caba, caya, etc.).
# 
# Redirect stdout with > to save to a file.
# Example:
# python automation/create_label_file.py ../wav ../general

import glob
import os.path
import numpy as np
import librosa
import sys
import pandas as pd
import json

show_histograms = True #use True to list histograms in the end

def write_label_dictionary(label_file, output_labels_dic_file):
    #read input file
    df = pd.read_csv(label_file) #read CSV label file using Pandas
    #initialize and pre-allocate space
    label_dict = {} #dictionary
    cur_label = 0
    for i, row in df.iterrows():
        label = row['annotation'].strip() #column annotation indicates the labels
        if label not in label_dict:
            label_dict[label] = cur_label
            cur_label += 1
    a_file = open(output_labels_dic_file, "w")
    json.dump(label_dict, a_file)
    a_file.close()

'''
Find one substring from given list of substrings in a string
'''
def find_substring_in_string(string, substrings_list):
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
    expected_Fs = 44100 #expected sample rate (all files should have it)
    #note: the label calla is the same as caya
    labels = ["caba","cada","caga","caca","capa","casa","cata","caya"] #,"NOISE"]
    #speakers = ["Mots","D2","Fuyi","Idir","Joel","Ricardo2","Tini"] #old
    speakers = ["David","Fuyi","Idir","Joel","JoseLuiz","Ricardo","Tini"]
    header = "filename,annotation,offset,starts,stops,speaker" #assumed header for labels

    histogram = np.zeros((len(labels),),dtype='int')
    speaker_histogram = np.zeros((len(speakers),),dtype='int')

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
        #find the label
        #note that this_label_index is not the integer that will represent the given class
        #this integer will be defined later, by the dictionary
        this_label, this_label_index = find_substring_in_string(filename_without_extension, labels)
        histogram[this_label_index] += 1

        #find the speaker
        this_speaker, this_speaker_index = find_substring_in_string(filename_without_extension, speakers)
        speaker_histogram[this_speaker_index] += 1

        #It is assumed:    
        #header = "filename,annotation,offset,starts,stops" #assumed header for labels
        #print(filename,",",this_label,",",this_label_index,",0,0,",x.shape[0],",",this_speaker,sep='')
        output_string = filename + "," + this_label + ",0,0," + str(x.shape[0]) + "," + this_speaker
        fp.write(output_string)
        fp.write('\n')

    if show_histograms: #enable to see the histogram
        #Histogram:
        #['caba', 'cada', 'caga', 'caca', 'capa', 'casa', 'calla', 'cata', 'caya']
        #[15 16 16 22 15 15  3 13 12]    
        print("Label Histogram:")
        print(labels)
        print(histogram)
        print("Speaker Histogram:")
        print(speakers)
        print(speaker_histogram)

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
    write_label_dictionary(label_file, output_labels_dic_file)
    print("Wrote file", output_labels_dic_file)
