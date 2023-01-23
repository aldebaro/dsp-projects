'''
This script aims to run all steps for a given experiment.
It assumes that the wav file is at the parent folder of the folder in which this
script is being executed.
The script can be executed on both Windows and Linux / MacOS.

Used 
https://stackoverflow.com/questions/89228/how-do-i-execute-a-program-or-call-a-system-command
https://stackoverflow.com/questions/44758588/running-python-script-in-interactive-python-prompt-and-keep-the-variables
and
https://docs.python.org/3/library/subprocess.html#subprocess.run
'''
import subprocess
import os
import sys

########### Configuration ###########
EXPERIMENT = 1 #choose the experiment number as an identifier to help organizing things
USE_MEYER_DATASET = False #use True to enable adopting waveform datasets for whistled speech (files JoseLuiz_caba.wav, etc.)
if USE_MEYER_DATASET:
    NUM_CLASSES = 8 #there are 8 speakers in Prof. Meyer's dataset
else: #choose the correct value for your dataset below
    NUM_CLASSES = 10 #number of classes (labels). Use 10 if the task is digits recognition
FEATURES = "mel" #use "mel", "magnasco" or "stft"
NORMALIZATION = "minmax" #use "minmax", "maggie", "std_freq" or "none"
USE_LOGDOMAIN = "--log_domain" #use "--log_domain" or an empty string ""
#each waveform file will be represented by a fature matrix of dimension D x T
D = 18 #number of frequency points in feature matrix of dimension D x T
T = 30 #number of time points in feature matrix of dimension D x T
LATENT_DIM = 3 #dimension of latent space (after transformation)
SUPER_EPOCHS = 2 #number of iterations of super loop
TRIPLET_EPOCHS = 10 #number of iterations when training triplets
EPOCHS = 10 #number of iterations when training neural network
LATENT_EPOCHS = 10 #number of iterations when training neural network that imposes latent space
NUM_TRIPLETS = 300 #number of triplets
EXPERIMENT_ID = str(EXPERIMENT) + FEATURES + "D" + str(D) + "T" + str(T) #experiment identifier
OUTPUT_DIR= "../outputs/" + EXPERIMENT_ID #output folder (will be created) to store output files
if True: #True if it is the first time you run the frontend
    OUTPUT_ID= FEATURES + "D" + str(D) + "T" + str(T)
    #list with all commands you want to execute from the list all_scripts to be defined below
    all_scripts_indices = [0,1,2,3,4,5,6,7] #execute all commands
    #all_scripts_indices = [7] #execute only command 7
else: #In case you have already executed the frontend and cut features
    #with signal_statistics.py to find Dmin and Dmax and later use cut_frequencies.py
    #For instance:
    #python feature-extraction\signal_statistics.py --input_file ..\outputs\4magnascoD370T120\features_magnascoD370T120.hdf5   
    #and then
    #python feature-extraction\cut_frequencies.py --input_file ..\outputs\4magnascoD370T120\features_magnascoD370T120.hdf5 --input_folder ..\outputs\4magnascoD370T120\features_magnascoD370T120\features_no_resizing --Dmin 10 --Dmax 200
    Dmin=80
    Dmax=265
    OUTPUT_ID= FEATURES + "D" + str(D) + "T" + str(T) + "_cutD" + str(Dmin) + "_" + str(Dmax)
    #list with all commands you want to execute from the list all_scripts to be defined below
    #all_scripts_indices = [2,3,4,5,6,7]
    all_scripts_indices = [5]

########### You probably do not need to change the code below #############

INPUT_FEATURES= OUTPUT_DIR + "/features_" + OUTPUT_ID + ".hdf5"
ENCODER_MODEL = OUTPUT_DIR + "/features_" + OUTPUT_ID + "/encoder_models/encoder.h5"
INPUT_LATENT = OUTPUT_DIR + "/features_" + OUTPUT_ID + "/encoder_models/latent_vectors.h5"
INPUT_LATENT_CSV = OUTPUT_DIR + "/features_" + OUTPUT_ID + "/encoder_models/latent_vectors.csv"
LATENT_MODEL = OUTPUT_DIR + "/features_" + OUTPUT_ID + "/encoder_models/best_model.hdf5"
#LATENT_MODEL = OUTPUT_DIR + "/features_" + OUTPUT_ID + "/encoder_models/last_model.hdf5"

def run_my_command(cmd):
    print("######### " + cmd + " #############")
    #subprocess.call(cmd, shell=True,  check=True)
    subprocess.run(cmd, shell=True, check=True)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR,exist_ok=True) #create folder if it does not exist
    print("Created output folder",OUTPUT_DIR)

#save the script with configuration in the output folder for reproducibility
#make sure the file separator is fine for the given OS
if os.name == 'nt':
    this_output_dir = OUTPUT_DIR.replace("/","\\")
    command = "copy " + sys.argv[0] + " " + this_output_dir
else: #Linux
    this_output_dir = OUTPUT_DIR.replace("\\","/")
    command = "cp " + sys.argv[0] + " " + this_output_dir
if os.path.exists(os.path.join(this_output_dir,"execute_all.py")):
    print("ERROR!", os.path.join(this_output_dir,"execute_all.py"), "already exists!")
else: #do not overwrite
    os.system(command)
    print("Executed: ", command)

if USE_MEYER_DATASET:
    cmd0="python automation/create_label_file.py ../wav ../general"
else:
    #assume ../wav has the contents of the "recording" folder of the FSDD dataset
    #https://github.com/Jakobovski/free-spoken-digit-dataset
    cmd0="python automation/fsdd_dataset_create_label_file.py ../wav ../general"
cmd1="python feature-extraction/general_frontend.py --D " + str(D) + " --T " + str(T) + " --output_dir " + OUTPUT_DIR + " --features " + FEATURES + " --normalization " + NORMALIZATION + " --save_plots" + " " + USE_LOGDOMAIN
cmd2="python machine-learning/fixed_dim_encoder_clusterer.py --input_file " + INPUT_FEATURES + " --triplet_epochs " + str(TRIPLET_EPOCHS) + " --epochs " + str(EPOCHS) + " --num_triplets " + str(NUM_TRIPLETS) + " --super_epochs " + str(SUPER_EPOCHS) + " --latent_dim " + str(LATENT_DIM) + " --num_classes " + str(NUM_CLASSES)
cmd3="python machine-learning/write_latent_vectors.py --input_file "  + INPUT_FEATURES + " --input_model " + ENCODER_MODEL
cmd4="python machine-learning/train_dnn_latent_space.py --test_indices_exist --input_file " + INPUT_LATENT + " --epochs " + str(LATENT_EPOCHS) + " --num_classes " + str(NUM_CLASSES)
cmd5="python machine-learning/evaluate_classifier.py --test_indices_exist --input_file " + INPUT_LATENT + " --input_model " + LATENT_MODEL
cmd6="python machine-learning/other_classifiers.py --test_indices_exist --input_file " + INPUT_LATENT
if LATENT_DIM == 3:
    cmd7="python machine-learning/plot_3d_tsne.py --input_file " + INPUT_FEATURES + " --input_tsne_file " + INPUT_LATENT_CSV
else:
    cmd7="" #plot_3d_tsne.py is only for 3D data

#note that the commands that will be effectively executed have been specified
# in the list all_scripts_indices, previously defined
all_scripts=[cmd0, cmd1, cmd2, cmd3, cmd4, cmd5, cmd6, cmd7]

#run all defined scripts
for i in range(len(all_scripts_indices)):
    this_index = all_scripts_indices[i]
    print("Script " + str(this_index) + " ****************************")
    cmd = all_scripts[this_index]
    #execute configuration script
    run_my_command(cmd)

print("Happy end!")