'''
Plot t-SNE or latent vectors in 3D.
'''
import pandas
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from ml_util import read_instances_from_file
import json
import argparse
import os

LABEL_SHIFT = 0 #if want, use a shift in pixels to better display the labels

def get_all_speakers(dataframe_all_files):    
    return dataframe_all_files['speaker'].unique()

'''
Plot a special figure per speaker. Show the names only for the files corresponding
to the given speaker.
'''
def plot3d_for_each_speaker(x,y,z,y_label,reverse_label_dict,test_indices,df_all_files,output_dir):
    all_speakers = get_all_speakers(df_all_files)
    num_speakers = len(all_speakers)
    for i in range(num_speakers):
        speaker = all_speakers[i]
        df_speaker = df_all_files.loc[df_all_files['speaker']==speaker]

        speaker_indices = df_speaker.index.values.tolist()

        filenames_per_index = df_speaker["filename"].tolist()
        fig_title = speaker
        plot3d_with_colorful_labels(x,y,z,y_label,reverse_label_dict,test_indices,speaker_indices,filenames_per_index,fig_title,output_dir)

'''
Plot examples with colors indicating the different labels (one color per word).
'''
def plot3d_with_colorful_labels(x,y,z,y_label,reverse_label_dict,test_indices,speaker_indices,filenames_per_index,fig_title,output_dir):
    num_examples = len(x)
    # setup the plot
    fig, ax = plt.subplots(1,1, figsize=(50,50))
    ax = fig.add_subplot(projection='3d')

    # define the colormap
    cmap = plt.cm.jet
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

    # define the bins and normalize
    bounds = np.linspace(0,num_classes,num_classes+1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    # indicate the test examples
    if True:
        this_x = x[test_indices]
        this_y = y[test_indices]
        this_z = z[test_indices]
        scat = ax.scatter(this_x,this_y,this_z,marker='o',s=180,alpha=0.08)

    # indicate the indices
    num_speaker_indices = len(speaker_indices)
    print("speaker_indices", speaker_indices) #print list
    for j in range(num_speaker_indices):
        i = speaker_indices[j]
        this_x = x[i]
        this_y = y[i]
        this_z = z[i]
        this_label = os.path.splitext(filenames_per_index[j])[0]
        #filenames_per_index
        ax.text(this_x-LABEL_SHIFT, this_y+LABEL_SHIFT, this_z+LABEL_SHIFT, this_label, (0,1,1))

    # make the scatter
    for i in range(num_classes):
        indices_of_class = (y_label == i)
        this_x = x[indices_of_class]
        this_y = y[indices_of_class]
        this_z = z[indices_of_class]
        label = str(i) + "_" + reverse_label_dict[i]
        #scat = ax.scatter(x,y,c=tag,s=np.random.randint(100,500,N),cmap=cmap,     norm=norm)
        #scat = ax.scatter(x,y,c=tag,s=100,alpha=0.6,cmap=cmap,norm=norm,marker='x')
        scat = ax.scatter(this_x,this_y,this_z,c=i*np.ones((len(this_x),)),cmap=cmap,norm=norm,marker='x',label=label)
        #ax.annotate(label, (this_x[0]-LABEL_SHIFT, this_y[0]+LABEL_SHIFT, this_z[0]+LABEL_SHIFT))
        #from https://matplotlib.org/stable/gallery/mplot3d/text3d.html
        #ax.text(this_x[0]-LABEL_SHIFT, this_y[0]+LABEL_SHIFT, this_z[0]+LABEL_SHIFT, label, (0,1,1))
        #ax.text(this_x[0]-LABEL_SHIFT, this_y[0]+LABEL_SHIFT, this_z[0]+LABEL_SHIFT, label, (0,1,1))
        #scat = ax.scatter(x,y)

    # create the colorbar
    #cb = plt.colorbar(scat, spacing='proportional',ticks=bounds)
    #cb.set_label('Custom cbar')
    ax.set_title(fig_title)

    #ax.legend()
    out_file = os.path.join(output_dir, fig_title + ".png")
    plt.savefig(out_file)
    print('Wrote file', out_file)
    #plt.show()
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--general_dir',help='folder with wavs_labels.csv and labels_dictionary.json files',default='../general')
    parser.add_argument('--test_indices',help='file with test indices (default: test_indices.csv at the same folder of the t-SNE result)')
    parser.add_argument('--output_dir',help='output folder where the PNGs will be saved (default: the same of the input t-SNE result)')    
    #required arguments    
    parser.add_argument('--input_tsne_file',help='input file with t-SNE result or latent vectors if dimension = 3 (e.g. output_tsne.csv)',required=True)
    parser.add_argument('--input_file',help='input file with features (to read the correct output y)',required=True)
    parser.add_argument('--num_classes',type=int, help='number of classes (labels)', default=8)
    args = parser.parse_args()

    general_dir = args.general_dir
    output_dir = args.output_dir
    num_classes = args.num_classes
    label_file = os.path.join(general_dir, "wavs_labels.csv")
    labels_dic_file = os.path.join(general_dir, "labels_dictionary.json")

    df_all_files = pandas.read_csv(label_file)

    tsne_data_file = args.input_tsne_file
    test_indices_file = args.test_indices
    instances_file = args.input_file #r'justsounds_todoscorrigidos_normalizados\spectrogramD50T30.hdf5'
    #labels_dic_file = r'justsounds_todoscorrigidos_normalizados\labels_dictionary.json'
    if test_indices_file == None:
        folder = os.path.dirname(tsne_data_file)
        test_indices_file = os.path.join(folder, "test_indices.csv")

    if output_dir == None:
        output_dir = os.path.dirname(tsne_data_file)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir,exist_ok=True)
        print("Created folder", output_dir)

    print("I am reading the following file:")
    print("test_indices_file=",test_indices_file)
    test_indices = np.genfromtxt(test_indices_file, delimiter=',')
    test_indices = test_indices.astype(int)

    df = pandas.read_csv(tsne_data_file, header=None)
    tsne_data = df.to_numpy()
    print(tsne_data.shape)
    num_examples, dimension = tsne_data.shape
    if dimension != 3:
        print("ERROR! This code assumes 3D data")
        exit(-1)

    a_file = open(labels_dic_file, "r")
    label_dict_str = a_file.read()
    print("Just read file", labels_dic_file)
    #https://appdividend.com/2022/01/29/how-to-convert-python-string-to-dictionary/
    label_dict = json.loads(label_dict_str)
    reverse_dict = dict([(v, k) for k, v in label_dict.items()]) #reverse dictionary    
    #print('reversed label_dict=',reverse_dict)

    #X is T x D matrix
    X, y = read_instances_from_file(instances_file)
    y = y.astype(int) #convert output labels to integers
    plot3d_for_each_speaker(tsne_data[:,0],tsne_data[:,1],tsne_data[:,2],y,reverse_dict,test_indices,df_all_files,output_dir)