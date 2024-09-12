import h5py
import os
import numpy as np
import pdb
import tqdm
import cv2

def process_data(ori_path, save_path, start_idx=0):
    h5py_path = os.path.join(save_path, 'h5py')
    if not os.path.exists(h5py_path): os.mkdir(h5py_path)
    cam_high_path = os.path.join(save_path, 'cam_high')
    if not os.path.exists(cam_high_path): os.mkdir(cam_high_path)
    cam_left_wrist_path = os.path.join(save_path, 'cam_left_wrist')
    if not os.path.exists(cam_left_wrist_path): os.mkdir(cam_left_wrist_path)
    cam_right_wrist_path = os.path.join(save_path, 'cam_right_wrist')
    if not os.path.exists(cam_right_wrist_path): os.mkdir(cam_right_wrist_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    for file_name in tqdm.tqdm(sorted(os.listdir(ori_path))[start_idx:]):
        if '.hdf5' not in file_name: continue
        read_h5py = h5py.File(os.path.join(ori_path, file_name), 'r')
        write_h5py = h5py.File(os.path.join(h5py_path, file_name),'w')
        cam_high_out = cv2.VideoWriter(os.path.join(cam_high_path, '{}.mp4'.format(file_name.split('.')[0])), fourcc, 20.0, (640, 480), True)
        cam_left_wrist_out = cv2.VideoWriter(os.path.join(cam_left_wrist_path, '{}.mp4'.format(file_name.split('.')[0])), fourcc, 20.0, (640, 480), True)
        cam_right_wrist_out = cv2.VideoWriter(os.path.join(cam_right_wrist_path, '{}.mp4'.format(file_name.split('.')[0])), fourcc, 20.0, (640, 480), True)

        write_h5py['action'] = read_h5py['action'][:]
        write_h5py['base_action'] = read_h5py['base_action'][:]
        write_h5py['/observations/effort'] = read_h5py['/observations/effort'][:]
        write_h5py['/observations/qpos'] = read_h5py['/observations/qpos'][:]
        write_h5py['/observations/qvel'] = read_h5py['/observations/qvel'][:]
        cam_high_frames = read_h5py['/observations/images/cam_high'][:]
        for frame in cam_high_frames:
            frame_bgr = np.ascontiguousarray(frame[:, :, ::-1])
            cam_high_out.write(frame_bgr)
        cam_left_wrist_frames = read_h5py['/observations/images/cam_left_wrist'][:]
        for frame in cam_left_wrist_frames:
            frame_bgr = np.ascontiguousarray(frame[:, :, ::-1])
            cam_left_wrist_out.write(frame_bgr)
        cam_right_wrist_frames = read_h5py['/observations/images/cam_right_wrist'][:]
        for frame in cam_right_wrist_frames:
            frame_bgr = np.ascontiguousarray(frame[:, :, ::-1])
            cam_right_wrist_out.write(frame_bgr)

        read_h5py.close()
        write_h5py.close()
        cam_high_out.release()
        cam_left_wrist_out.release()
        cam_right_wrist_out.release()
        
    print('Done!')

if __name__ == '__main__':
    process_data(ori_path = '/home/cvte/twilight/home/data/grasp_toy/src', save_path = '/home/cvte/twilight/home/data/grasp_toy/convert', start_idx = 0)