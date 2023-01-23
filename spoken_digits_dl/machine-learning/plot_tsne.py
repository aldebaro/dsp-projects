'''
Plot t-SNE
'''
import pandas
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from ml_util import read_instances_from_file
import json
import argparse
import os

N = 8 # Number of labels
LABEL_SHIFT = 30

def plot_with_colorful_labels(x,y,tag,reverse_label_dict):
    print('x',x.shape)
    print('y',y.shape)
    print('tag',tag.shape)
    # setup the plot
    fig, ax = plt.subplots(1,1, figsize=(6,6))
    # define the data
    #x = np.random.rand(1000)
    #y = np.random.rand(1000)
    #tag = np.random.randint(0,N,1000) # Tag each point with a corresponding label    

    # define the colormap
    cmap = plt.cm.jet
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

    # define the bins and normalize
    bounds = np.linspace(0,N,N+1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    # make the scatter
    for i in range(N):
        indices_of_class = (tag == i)
        this_x = x[indices_of_class]
        this_y = y[indices_of_class]
        label = str(i) + "_" + reverse_label_dict[i]
        #scat = ax.scatter(x,y,c=tag,s=np.random.randint(100,500,N),cmap=cmap,     norm=norm)
        #scat = ax.scatter(x,y,c=tag,s=100,alpha=0.6,cmap=cmap,norm=norm,marker='x')
        scat = ax.scatter(this_x,this_y,c=i*np.ones((len(this_x),)),s=100,cmap=cmap,norm=norm,marker='x',label=label)
        ax.annotate(label, (this_x[0]-LABEL_SHIFT, this_y[0]+LABEL_SHIFT))
        #scat = ax.scatter(x,y)
    # create the colorbar
    cb = plt.colorbar(scat, spacing='proportional',ticks=bounds)
    cb.set_label('Custom cbar')
    ax.set_title('Clustered words using Triplet loss + t-SNE')

    #ax.legend()
    out_file = './final_clustering.png'
    plt.savefig(out_file)
    print('Wrote file', out_file)
    plt.show()
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--general_dir',help='folder with wavs_labels.csv and labels_dictionary.json files',default='../general')
    #required arguments    
    parser.add_argument('--input_tsne_file',help='input file with t-SNE result (e.g. output_tsne.csv)',required=True)
    parser.add_argument('--input_file',help='input file with features (to read the correct output y)',required=True)
    args = parser.parse_args()

    general_dir = args.general_dir
    #label_file = os.path.join(general_dir, "wavs_labels.csv")
    labels_dic_file = os.path.join(general_dir, "labels_dictionary.json")

    tsne_data_file = args.input_tsne_file
    instances_file = args.input_file #r'justsounds_todoscorrigidos_normalizados\spectrogramD50T30.hdf5'
    #labels_dic_file = r'justsounds_todoscorrigidos_normalizados\labels_dictionary.json'

    df = pandas.read_csv(tsne_data_file, header=None)
    tsne_data = df.to_numpy()
    print(tsne_data.shape)

    a_file = open(labels_dic_file, "r")
    label_dict_str = a_file.read()
    #https://appdividend.com/2022/01/29/how-to-convert-python-string-to-dictionary/
    label_dict = json.loads(label_dict_str)
    reverse_dict = dict([(v, k) for k, v in label_dict.items()]) #reverse dictionary    
    print('reversed label_dict=',reverse_dict)

    #X is T x D matrix
    X, y = read_instances_from_file(instances_file)
    y = y.astype(int) #convert output labels to integers
    plot_with_colorful_labels(tsne_data[:,0],tsne_data[:,1],y,reverse_dict)