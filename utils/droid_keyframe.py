import h5py
import numpy as np
import pdb
import tqdm
import os
import cv2
import pickle

def droid_keyframe(root_path, vis_keyframe):
    h5py_filename_list = sorted(os.listdir(os.path.join(root_path, 'h5py')))
    info_root = os.path.join(root_path, 'info')

    gripper_position_threshold = 0.5

    keyframe_gap = 20   # If there is a keyframe, no new keyframe in the following 20 frames.
    static_horizon = 6
    static_offset_threshold = 0.01

    max_keyframe_num = 0
    for h5py_filename in tqdm.tqdm(h5py_filename_list):
        h5py_file = h5py.File(os.path.join(root_path, 'h5py', h5py_filename), 'r')
        cartesian_position = h5py_file['observation']['cartesian_position'][:] # Left shape: (T, 6). The first 3 numbers are xyz and the last 3 numbers are possibly roll, pitch, yaw.
        gripper_position = h5py_file['observation']['gripper_position'][:] # Left shape: (T, 1), range: (0, 1).

        keyframe_idxs = []
        last_frame_static_flag = False
        last_keyframe_idx = None
        for idx in range(1, cartesian_position.shape[0]):
            if last_keyframe_idx != None and idx < last_keyframe_idx + keyframe_gap: continue
            # Gripper position crosses the threshold.
            if (gripper_position[idx - 1] - gripper_position_threshold) * (gripper_position[idx] - gripper_position_threshold) < 0:
                keyframe_idxs.append(idx)
                last_keyframe_idx = idx
            # Gripper keeps static for a while.
            xyz_moment = cartesian_position[:, :3]
            if idx < cartesian_position.shape[0] - static_horizon:
                if np.max(np.linalg.norm(xyz_moment[idx + 1 : idx + static_horizon] - xyz_moment[idx : idx + 1], axis = 1)) < static_offset_threshold:
                    if last_frame_static_flag == False: 
                        keyframe_idxs.append(idx)
                        last_keyframe_idx = idx
                    last_frame_static_flag = True
                else:
                    last_frame_static_flag = False
        h5py_file.close()
        if len(keyframe_idxs) > max_keyframe_num:
            max_keyframe_num = len(keyframe_idxs)

        info_dict = dict(keyframe_idxs = keyframe_idxs)
        with open(os.path.join(info_root, h5py_filename[:-5] + '.pkl'), 'wb') as f:
            pickle.dump(info_dict, f)

        if vis_keyframe:
            read_cap = cv2.VideoCapture(os.path.join(root_path, 'exterior_image_1_left', h5py_filename[:-5] + '.mp4'))
            width, height = int(read_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(read_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = read_cap.get(cv2.CAP_PROP_FPS)
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            write_writter = cv2.VideoWriter('vis.mp4', fourcc, fps, (width, height))
            cnt = 0
            print("Key Frame index list: {}".format(keyframe_idxs))
            while read_cap.isOpened():
                ret, frame = read_cap.read()
                if not ret: break
                if cnt in keyframe_idxs:
                    cv2.putText(frame, "Key Frame", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
                    for i in range(10):
                        write_writter.write(frame)
                else:
                    write_writter.write(frame)
                cnt += 1
            read_cap.release()
            write_writter.release()
            pdb.set_trace()
    print("Max keyframe number: {}".format(max_keyframe_num))

if __name__ == '__main__':
    droid_keyframe(
        root_path = '/home/cvte/twilight/home/data/droid_h5py',
        vis_keyframe = False
    )
