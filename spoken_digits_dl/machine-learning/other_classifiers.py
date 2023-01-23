'''
Train and test several classifiers.
# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler and at UFPA
# License: BSD 3 clause
'''

import numpy as np
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import pickle

import argparse
from ml_util import *

#define names and their respective classifier below
names = ["BaggingKNNs",
            "BaggingSVMs",
            "BaggingDecisionTrees",
             "NaiveBayes",
             "DecisionTree",
             "RandomForest",
             "AdaBoost",
             "LinearSVM", 
             "RBFSVM", 
             "GaussianProcess",
             #"NeuralNet",
             "QDA", 
             "NearestNeighbors"]

classifiers = [
        BaggingClassifier(base_estimator=KNeighborsClassifier(5),n_estimators=10, random_state=0),
        BaggingClassifier(base_estimator=SVC(),n_estimators=10, random_state=0),
        BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=10),n_estimators=20, random_state=0),
        GaussianNB(),
        DecisionTreeClassifier(max_depth=10),
        RandomForestClassifier(max_depth=10, n_estimators=30),
        AdaBoostClassifier(),
        LinearSVC(dual = True), #linear SVM (maximum margin perceptron)
        SVC(gamma=1, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        #MLPClassifier(alpha=0.1, max_iter=5000),
        QuadraticDiscriminantAnalysis(),
        KNeighborsClassifier(3)]


def train_classifier(clf_name, X, y, num_classes):
    """
    Parameters
    ==========
    clf_name: str
        Classifier name to be selected
    X:
    y:
    num_classes: int
        Total number of classes
    """
    assert(clf_name in names)

    clf_ind = names.index(clf_name)
    clf = classifiers[clf_ind]
    clf.fit(X, y)

    return clf

def train_and_test_all_classifiers(X_train, y_train, X_test, y_test, output_dir):

    test_errors = list()
    for name in names:
        print("###### Training classifier: ", name)
        output_file = os.path.join(output_dir, name + ".pkl")

        clf = train_classifier(name, X_train, y_train, np.max(y))

        #save file
        #https://stackoverflow.com/questions/10592605/save-classifier-to-disk-in-scikit-learn
        with open(output_file, 'wb') as fid:
            pickle.dump(clf, fid)

        clf.fit(X_train, y_train)

        pred_train = clf.predict(X_train)
        print('Prediction accuracy for the train dataset:\t {:.2%}'.format(
            metrics.accuracy_score(y_train, pred_train)
        ))

        pred_test = clf.predict(X_test)
        acc_test = metrics.accuracy_score(y_test, pred_test)
        print('Prediction accuracy for the test dataset:\t {:.2%}'.format(acc_test))

        error_test = 1.0 - acc_test
        print('Misclassification error for the test dataset:\t {:.2%}'.format(
            error_test
        ))

        print('\n')

        test_errors.append(error_test)
    return test_errors

if __name__ == '__main__':
    print("=====================================")
    print("Train and test classifiers")

    parser = argparse.ArgumentParser()
    parser.add_argument('--general_dir',help='folder with wavs_labels.csv and labels_dictionary.json files',default='../general')
    parser.add_argument('--show_plot', action='store_true') #default is false
    parser.add_argument('--output_dir',help='output folder (default is the folder of the input file)')
    parser.add_argument('--disjoint_speakers', action='store_true') #default is false
    parser.add_argument('--test_indices_exist', help='a file called test_indices.csv in the input file folder must be used for defining the test set', action='store_true') #default is false    
    #required arguments    
    parser.add_argument('--input_file',help='input file with features',required=True)
    args = parser.parse_args()

    test_indices_exist = args.test_indices_exist
    show_images = args.show_plot #show images
    instances_file = args.input_file
    general_dir = args.general_dir
    label_file = os.path.join(general_dir, "wavs_labels.csv")
    labels_dic_file = os.path.join(general_dir, "labels_dictionary.json")
    use_disjoint_speakers = args.disjoint_speakers

    if args.output_dir == None:
        #create a default name
        output_folder = os.path.dirname(instances_file)
    else:
        output_folder = args.output_dir

    if not os.path.exists(output_folder):
        os.makedirs(output_folder,exist_ok=True) #create folder if it does not exist
        print("Created output folder",output_folder)

    #read the labels_file
    df_input_data = pd.read_csv(label_file) #read CSV label file using Pandas
    num_examples = len(df_input_data)
    if num_examples != 127:
        print("WARNING!!!! num_examples=", num_examples)

    #save this script in the output folder for reproducibility
    if os.name == 'nt':
        command = "copy " + sys.argv[0] + " " + output_folder
    else: #Linux
        command = "cp " + sys.argv[0] + " " + output_folder
    os.system(command)
    print("Executed: ", command)

    X, y = read_instances_from_file(instances_file)
    y = y.astype(int) #convert output labels to integers
    #num_examples, D, T = X.shape

    if len(X.shape) == 3:
        num_examples, T, D = X.shape #we use the transpose
        num_rows = T
        num_columns = D
        #Convert matrices in X into vectors
        X = np.reshape(X, (num_examples, -1))
    else: #latent space
        num_examples, latent_dim = X.shape #we use the transpose
        num_rows = latent_dim
        num_columns = 1

    a_file = open(labels_dic_file, "r")
    label_dict_str = a_file.read()
    #https://appdividend.com/2022/01/29/how-to-convert-python-string-to-dictionary/
    label_dict = json.loads(label_dict_str)
    print("labels_dict", label_dict)
    print("labels_dict", type(label_dict))

    all_scores = list()
    #split into train and test correctly
    if use_disjoint_speakers:
        speaker_indices_dic, all_speakers = split_into_train_test_disjoint_speakers(df_input_data)
        num_speakers = len(all_speakers)
        for i in range(num_speakers):
            test_speaker = all_speakers[i]
            print("############## Training model for test speaker", test_speaker)
            X_train, X_test, y_train, y_test, train_indices, test_indices = compose_train_test_disjoint_speakers(X, y, speaker_indices_dic, all_speakers, test_speaker)
            score = train_and_test_all_classifiers(X_train, y_train, X_test, y_test, output_folder)
            all_scores.append(score)
        print(all_speakers)
        print('Test scores')
        print(all_scores)
    else:
        if test_indices_exist:
            temp_dir = os.path.dirname(instances_file)
            file_with_test_indices = os.path.join(temp_dir,"test_indices.csv")
            if not os.path.exists(file_with_test_indices):
                print("ERROR: could not find file", file_with_test_indices, "in folder", temp_dir)
                exit(-1)
            test_indices = np.genfromtxt(file_with_test_indices, delimiter=',')
            test_indices = test_indices.astype(int)
            file_with_train_indices = os.path.join(temp_dir,"train_indices.csv")
            train_indices = np.genfromtxt(file_with_train_indices, delimiter=',')                
            train_indices = train_indices.astype(int)
            print("train_indices", train_indices, type(train_indices))
            print("test_indices", test_indices, type(test_indices))
            y_train = y[train_indices]
            y_test = y[test_indices]
            #X_test = [ X[i] for i in test_indices ]
            #X_train = [ X[i] for i in train_indices ]
            X_train = X[train_indices]
            X_test = X[test_indices]
        else:
            test_fraction = 0.2
            X_train, X_test, y_train, y_test, train_indices, test_indices = split_into_train_test_mixed_speakers(X, y, test_fraction)
            file_with_test_indices = os.path.join(output_folder,"other_classifiers_test_indices.csv")
            np.savetxt(file_with_test_indices, test_indices, delimiter=",")
            file_with_train_indices = os.path.join(output_folder,"other_classifiers_train_indices.csv")
            np.savetxt(file_with_train_indices, train_indices, delimiter=",")

        score = train_and_test_all_classifiers(X_train, y_train, X_test, y_test, output_folder)
