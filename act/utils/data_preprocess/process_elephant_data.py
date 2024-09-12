import h5py
import os
import numpy as np
import pdb
import tqdm
import cv2

def process_data(ori_path, save_path, start_idx=0):
    h5py_path = os.path.join(save_path, 'h5py')
    if not os.path.exists(h5py_path): os.mkdir(h5py_path)
    cam_hand_path = os.path.join(save_path, 'hand')
    if not os.path.exists(cam_hand_path): os.mkdir(cam_hand_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    for file_name in tqdm.tqdm(sorted(os.listdir(ori_path))[start_idx:]):
        if '.hdf5' not in file_name: continue
        read_h5py = h5py.File(os.path.join(ori_path, file_name), 'r')
        write_h5py = h5py.File(os.path.join(h5py_path, file_name),'w')
        cam_hand_out = cv2.VideoWriter(os.path.join(cam_hand_path, '{}.mp4'.format(file_name.split('.')[0])), fourcc, 20.0, (640, 480), True)

        write_h5py['action'] = read_h5py['action'][:]
        write_h5py['/observations/qpos'] = read_h5py['/observations/qpos'][:]
        write_h5py['/observations/qvel'] = read_h5py['/observations/qvel'][:]
        hand_frames = read_h5py['/observations/images/hand'][:]
        for frame in hand_frames:
            frame_bgr = np.ascontiguousarray(frame)
            cam_hand_out.write(frame_bgr)

        read_h5py.close()
        write_h5py.close()
        cam_hand_out.release()
        
    print('Done!')

if __name__ == '__main__':
    process_data(ori_path = '/media/cvte/zli/data/elephant_grasp/src', save_path = '/media/cvte/zli/data/elephant_grasp/convert', start_idx = 0)