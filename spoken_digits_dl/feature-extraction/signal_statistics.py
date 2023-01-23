'''
Get some statistics about the features.
'''
import argparse
from utils_frontend import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file',help='input file with time-freq representation',required=True)
    parser.add_argument('--show_plot', action='store_true') #default is false
    args = parser.parse_args()

    intputHDF5FileName = args.input_file
    X, _ = read_instances_from_file(intputHDF5FileName)
    occurrences_above_reference = values_above_threshold_per_frequency(X)

    print("For each frequency dimension, number of occurrences above threshold")
    print(occurrences_above_reference)
    print("The frequency indices below are non-informative and could be removed by cut_frequencies.py:")
    print(np.where(occurrences_above_reference==0))

    if args.show_plot:
        for i in range(X.shape[0]):
            spec = X[i]
            plot_feature(spec,"Features")        
            print("Range = ", np.min(spec), np.max(spec))

