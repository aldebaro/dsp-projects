#https://pypi.org/project/praat-textgrids/
# Find all .wav files in a directory, pre-emphasize and save as new .wav and .aiff file
import parselmouth
import textgrids
import numpy as np

import glob
import os.path
import os

output_dir = "./outputs"
if os.path.exists(output_dir):
    print(output_dir,"already exists")
else:
    os.makedirs(output_dir)

print("aqui")

#for wave_file in glob.glob("./audio/*.wav"): #find all files with extension wav in current folder

for wave_file in glob.glob("./*.wav"): #find all files with extension wav in current folder
    print("Processing {}...".format(wave_file))
    s = parselmouth.Sound(wave_file) #open wav
    #print("Original audio:", s) #enable if want to see information about the waveform
    path_and_file_name = os.path.splitext(wave_file)[0] #discard extension
    file_name = os.path.basename(path_and_file_name)
    #extract waveform from Sound object 
    #https://parselmouth.readthedocs.io/en/stable/examples/plotting.html    
    waveform = s.values.T #waveform is a numpy array, T uses the transpose
    #print(waveform.__class__.__name__) #if wants to find out the class of the object
    #print(waveform.shape)
    discretized_time = s.xs()
    Ts = discretized_time[1] - discretized_time[0]  #sampling interval (sec)
    Fs = 1/Ts
    print('sampling rate (Hz) =', Fs)

    #TextGrid is an collections.OrderedDict whose keys are tier names (strings)
    # and values are Tier objects. The constructor takes an optional filename 
    # argument for easy loading and parsing textgrid files.
    textgrid_file_name = path_and_file_name + '.TextGrid' #create file name    
    my_textgrid = textgrids.TextGrid(textgrid_file_name) #open TextGrid
    print("Opening file:", textgrid_file_name)    
    print(my_textgrid.keys()) #there is a single tier called "type" here
    print(my_textgrid.values())

    my_intervals=my_textgrid['type'] #list of objects of class Interval
    #print(my_tier.__class__.__name__) #https://stackoverflow.com/questions/510972/getting-the-class-name-of-an-instance    
    N = len(my_intervals) #number of intervals / tiers
    print("Found",N,"intervals")
    #go over all words and segment
    word_number = 1
    for i in range(N): #check the text:
        this_text = my_intervals[i].text
        this_text = this_text.transcode() #convert to string
        #print(this_text)
        if this_text == "word": #find the word            
            print("Found a word")
            start_sample = int(np.round(my_intervals[i].xmin / Ts))
            end_sample = int(np.round(my_intervals[i].xmax / Ts))
            print("Duration in seconds: ", my_intervals[i].dur)
            print("Duration in samples: ", np.round(my_intervals[i].dur / Ts))
            #convert this waveform segment into a Sound object and save to file    
            #https://stackoverflow.com/questions/57216060/how-to-calculate-audio-metrics-through-parselmouth-on-a-subsequence-of-audio
            #Constructor: https://parselmouth.readthedocs.io/en/stable/api_reference.html#parselmouth.Sound.__init__
            option_Sofis = 2
            if option_Sofis == 1:
                #NOT WORKING: need to learn how to use the constructor 
                #gives different results with the vector is transposed
                this_word_waveform = waveform[start_sample:end_sample]
                #print(this_word_waveform.T.shape)
                print(this_word_waveform.shape)
                this_segment = parselmouth.Sound(this_word_waveform, Fs, 0.0)
                print(this_segment.get_sampling_frequency())
            else: #second option (this one is working)
                this_segment = s.extract_part(my_intervals[i].xmin, my_intervals[i].xmax)
            #save file
            output_file_name = os.path.join(output_dir,file_name + "_sofia_" + str(word_number) + ".wav")
            print("Save file", output_file_name)
            #print("This audio:", this_segment) #enable to print info about the file
            this_segment.save(output_file_name, parselmouth.SoundFileFormat.WAV) # or parselmouth.SoundFileFormat.WAV instead of 'WAV'
            word_number += 1 #increment counter