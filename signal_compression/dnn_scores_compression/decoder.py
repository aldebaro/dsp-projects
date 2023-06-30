import scipy.io, os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.io import savemat


def main():
    example_idx = '01'
    split_idx = '01' # or 16
    
    frame_idx_str = '000'
    frame_idx_int = 0

    output_folder = f'./uncompressed/data/example_{example_idx}/scores_mat/split_{split_idx}/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for total_frames in range(300):
        
        # load file
        frame_data_compressed = scipy.io.loadmat(f'./compressed/data/example_{example_idx}/scores_mat/split_{split_idx}/scores_frame_{frame_idx_str}.mat')
        frame_data_compressed = np.squeeze(frame_data_compressed['data'])

        # put here your code to uncompress the scores
        # below is just a lazy uncompression to exemplify
        frame_data_uncompressed = np.zeros((256, 200, 272), np.float32)
        for kernel_idx in range(frame_data_compressed.shape[0]):
            for row in range(frame_data_compressed.shape[1]):
                for column in range(frame_data_compressed.shape[2]):
                    frame_data_uncompressed[kernel_idx, row, column] = frame_data_compressed[kernel_idx, row, column]
        
        # save file
        data_dic = {"data": frame_data_uncompressed, "label":frame_idx_str}
        savemat(f'{output_folder}/scores_frame_{frame_idx_str}.mat', data_dic)
        
        # update frame count
        frame_idx_int += 1
        if frame_idx_int < 10:
            frame_idx_str = '00' + str(frame_idx_int)
        elif frame_idx_int < 100:
            frame_idx_str = '0' + str(frame_idx_int)
        else:
            frame_idx_str = str(frame_idx_int)

if __name__ == "__main__":
    main()