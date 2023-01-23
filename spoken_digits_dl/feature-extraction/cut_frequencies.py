'''
Discards specified frequency ranges.
To learn about frequency ranges that are not useful for machine learning,
one can run, e.g.:
  python feature-extraction\signal_statistics.py --input_file ..\stftD128T400\stftD128T400.hdf5

Then, later, one can use this script:
  python feature-extraction\cut_features.py --input_file ..\stftD128T400\stftD128T400.hdf5 --Dmin 3 --Dmax 76
The output will use the same folder and similar name to the input file
'''
import argparse
from utils_frontend import *
import pickle
import glob
import os

'''
Read X from pickle files and create y.
X is a list with variable dimension arrays.
'''
def cut_frequencies_variable_duration_instances_from_files(input_folder, output_folder, Dmin, Dmax):
    os.makedirs(output_folder,exist_ok=True) #create folder if it does not exist
    iterator = glob.glob(os.path.join(input_folder,"*_fea.pkl"))
    num_fea_files = len(iterator)
    print("Found", num_fea_files, "files with suffix _fea.pkl in folder", input_folder)
    for input_feature_file in iterator:
        basename = os.path.basename(input_feature_file)
        output_feature_file = os.path.join(output_folder, basename)
        with open(input_feature_file, 'rb') as f:
            this_X = pickle.load(f)
        print("Read",input_feature_file, "initial shape=", this_X.shape)
        new_X = this_X[Dmin:Dmax+1,:] #dimension is D x T
        with open(output_feature_file, 'wb') as f:
            pickle.dump(new_X, f)
        print("Wrote",output_feature_file, "final shape=", new_X.shape)
        #print(type(label_dict_str), label_dict_str,"ajad", this_label)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--general_dir',help='folder with wavs_labels.csv and labels_dictionary.json files',default='../general')
    parser.add_argument('--Dmin',type=int,help='First index of chosen frequency range. Default is 0, the first in input file',default=0)
    parser.add_argument('--Dmax',type=int,help='Last index of chosen frequency range. Use -1 to indica the last index in input file (default)',default=-1)
    parser.add_argument('--output_folder',help='folder with feature output files. Default is to share parent with input folder.')
    #required arguments    
    parser.add_argument('--input_file',help='input file with time-freq (extension hdf5) representation',required=True)
    parser.add_argument('--input_folder',help='folder with variable-dimension feature input (extension pkl) files',required=True)
    args = parser.parse_args()
    #In case you have already executed the frontend and cut features
    #with signal_statistics.py to find Dmin and Dmax and later use cut_frequencies.py
    #For instance:
    #python feature-extraction\signal_statistics.py --input_file ..\outputs\4magnascoD370T120\features_magnascoD370T120.hdf5   
    #and then
    #python feature-extraction\cut_frequencies.py --input_file ..\outputs\4magnascoD370T120\features_magnascoD370T120.hdf5 --input_folder ..\outputs\4magnascoD370T120\features_magnascoD370T120\features_no_resizing --Dmin 10 --Dmax 200

    intputHDF5FileName = args.input_file
    general_dir = args.general_dir
    input_folder = args.input_folder
    output_folder = args.output_folder
    Dmin = args.Dmin
    Dmax = args.Dmax

    if output_folder == None:
      temp_folder = os.path.dirname(input_folder)
      last_folder = os.path.basename(os.path.normpath(input_folder))
      output_folder = os.path.join(temp_folder, last_folder + "_cutD" + str(Dmin) + "_" + str(Dmax))

    print("AAA", input_folder, output_folder)

    X, y = read_instances_from_file(intputHDF5FileName)

    (num_examples, T, D) = X.shape
    if Dmax == -1:
        Dmax = D-1

    newD = Dmax - Dmin + 1
    print("Original D=", D, ". New D=", newD)
    outputHDF5FileName = os.path.splitext(intputHDF5FileName)[0] + "_cutD" + str(Dmin) + "_" + str(Dmax) + ".hdf5"

    newX = X[:,:,Dmin:Dmax+1]

    write_instances_to_file(outputHDF5FileName, newX, y)
    print("Wrote file", outputHDF5FileName, "with data of dimension", newX.shape)

    cut_frequencies_variable_duration_instances_from_files(input_folder, output_folder, Dmin, Dmax)
