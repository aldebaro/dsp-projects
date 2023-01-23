import numpy as np
import bhtsne
import pandas
import argparse

MAX_ITER = 50000

parser = argparse.ArgumentParser()
parser.add_argument('--output_file',help='output CSV file with vectors in 2D',default='output_tsne.csv')
#required arguments    
parser.add_argument('--latent_vectors',help='input file latent_vectors.csv',required=True)
args = parser.parse_args()

latent_vectors_file = args.latent_vectors
output_txt_file = args.output_file

#df = pandas.read_csv("iris.csv")
#df = df.drop("species", axis=1)

#without header
df = pandas.read_csv(latent_vectors_file, header=None)
#print(df)
data = df.to_numpy()
print(data.shape)
#print(data[0])
#data = np.loadtxt("iris.csv", skiprows=1)

embedding_array = bhtsne.run_bh_tsne(data, initial_dims=data.shape[1], verbose=True, perplexity=8, max_iter=MAX_ITER)
print(embedding_array)

np.savetxt(output_txt_file, embedding_array, delimiter=',')
print("Wrote CSV file without header", output_txt_file)