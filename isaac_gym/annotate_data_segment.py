import cv2
import h5py
import pdb
import os
import numpy as np

def annotate_data_segment(root_path, start_idx = 0):
    task_max_seg = 3
    h5py_path = os.path.join(root_path, 'h5py')
    exterior1_path = os.path.join(root_path, 'exterior_camera1')
    h5py_name_list = sorted(os.listdir(h5py_path))
    for cnt in range(start_idx, len(h5py_name_list)):
        h5py_file_name = h5py_name_list[cnt]
        replay_flag = True
        while replay_flag:
            file_id = h5py_file_name.split('.')[0]
            exter1_file_path = os.path.join(exterior1_path, file_id + '.mp4')
            seg_id_list = annotate_one_segment(exter1_file_path)
            print(seg_id_list)
            key = input("Save or replay this video? (s: save, r: replay)")
            if key == 's':
                with h5py.File(os.path.join(h5py_path, h5py_file_name), 'r+') as h5py_file:
                    if 'seg_keyframe' in h5py_file.keys(): del h5py_file['seg_keyframe']
                    h5py_file['seg_keyframe'] = np.array(seg_id_list)
                    replay_flag = False
            elif key == 'r':
                replay_flag = True

def annotate_one_segment(video_path):
    print("Tackle video: {}".format(video_path))
    seg_id_list = []
    cap = cv2.VideoCapture(video_path)
    cap_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_cnt = 0
    while True:
        if frame_cnt >= cap_total_frames: break
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1280, 960))
        cv2.imshow('Vis', frame)
        frame_cnt += 1

        key = cv2.waitKey(50)
        if key == -1: continue
        number = chr(key)
        if number >= '0' and number <= '9':
            number = int(number)
            seg_id_list.append((frame_cnt, number))
            print(f"Status {number} keyframe: {frame_cnt}")
    cap.release()
    return seg_id_list


if __name__ == '__main__':
    root_path = '/home/cvte/twilight/data/isaac_multicolorbox'
    annotate_data_segment(root_path)
    