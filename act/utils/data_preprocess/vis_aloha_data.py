import numpy as np
import h5py
import pdb
import os
import cv2
import tqdm
import matplotlib.pyplot as plt

def vis_aloha_sim_h5py(h5py_file, save_folder):
    h5py_content = h5py.File(h5py_file, 'r')
    frames = np.array(h5py_content['observations']['images']['top'])
    
    if os.path.exists(save_folder) is False:
        os.makedirs(save_folder)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.join(save_folder, '{}.mp4'.format(h5py_file.rsplit('/', 1)[-1].split('.')[0])), fourcc, 20.0, (640, 480), True)

    for i in range(frames.shape[0]):
        bgr_frame = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)
        out.write(bgr_frame)

    out.release()

def vis_self_data(h5py_file, save_folder, start_frame, end_frame):
    h5py_content = h5py.File(h5py_file, 'r')

    hand_frames = np.array(h5py_content['observations']['images']['hand'][start_frame:end_frame])
    top_frames = np.array(h5py_content['observations']['images']['top'][start_frame:end_frame])

    if os.path.exists(save_folder) is False:
        os.makedirs(save_folder)
    hand_save_path = os.path.join(save_folder, 'hand_camera.mp4')
    top_save_path = os.path.join(save_folder, 'top_camera.mp4')

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    hand_out = cv2.VideoWriter(hand_save_path, fourcc, 10.0, (640, 480), True)
    top_out = cv2.VideoWriter(top_save_path, fourcc, 10.0, (640, 480), True)

    for i in tqdm.tqdm(range(hand_frames.shape[0])):
        hand_out.write(hand_frames[i])
        top_out.write(top_frames[i])

    hand_out.release()
    top_out.release()
    print('Done')

def vis_aloha_action(h5py_path, start_idx, sample_gap, vis_len):
    h5py_file = h5py.File(h5py_path)
    action = h5py_file['action'][:]
    h5py_file.close()
    left_action = action[:, :7]
    sample_left_action = left_action[start_idx: start_idx + vis_len : sample_gap]

    plt.figure(figsize=(10, 8))
    for i in range(sample_left_action.shape[1]):
        plt.subplot(sample_left_action.shape[1], 1, i + 1)
        plt.scatter(range(sample_left_action.shape[0]), sample_left_action[:, i], label=f'Dimension {i+1}')
        plt.legend(loc='upper right')
        plt.title(f'Joint {i+1}')

    plt.tight_layout()
    plt.savefig('vis.png')


if __name__ == '__main__':
    #vis_aloha_sim_h5py(h5py_file = '/home/twilight/home/data/aloha/sim_insertion_scripted/episode_0.hdf5', save_folder = '/home/twilight/home/data/aloha/vis')
    #h5py_file = '/home/twilight/home/data/own_data/episode_0.hdf5'
    #vis_self_data(h5py_file = h5py_file, save_folder = '/home/twilight/home/data/own_data/vis/episode_0', start_frame = 10, end_frame = 400)
    vis_aloha_action(h5py_path = '/home/agilex/twilight/data/aloha_beverage/aloha_beverage/h5py/episode_32.hdf5', 
                     start_idx = 1000,
                     sample_gap = 1,
                     vis_len = 100)