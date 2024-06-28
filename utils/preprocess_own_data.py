import h5py
import os
import numpy as np
import pdb

def process_data(ori_path, save_path):
    for file_name in sorted(os.listdir(ori_path)):
        if '.hdf5' not in file_name: continue
        read_file = h5py.File(os.path.join(ori_path, file_name), 'r')
        write_file = h5py.File(os.path.join(save_path, file_name),'w')
        write_file['action'] = np.array(read_file['action'])
        write_file['/observations/images/hand'] = np.array(read_file['/observations/images/hand'])
        write_file['/observations/images/top'] = np.array(read_file['/observations/images/top'])
        write_file['/observations/qpos'] = np.array(read_file['/observations/qpos'])[:, :4]
        read_file.close()
        write_file.close()

if __name__ == '__main__':
    process_data(ori_path = '/home/twilight/home/data/own_data/buf', save_path = '/home/twilight/home/data/own_data')