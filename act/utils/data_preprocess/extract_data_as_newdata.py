import h5py
import os
import numpy as np
import pdb
import cv2
import tqdm

class DataExtractor:
    def __init__(self, ori_root, tgt_root, seg_id_dict):
        self.ori_root = ori_root
        self.ori_h5py_path = os.path.join(self.ori_root, 'h5py')

        self.tgt_root = tgt_root
        self.tgt_h5py_path = os.path.join(self.tgt_root, 'h5py')
        if not os.path.exists(self.tgt_h5py_path): os.makedirs(self.tgt_h5py_path)
        self.cam_names = ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
        for cam_name in self.cam_names:
            tgt_cam_path = os.path.join(self.tgt_root, cam_name)
            if not os.path.exists(tgt_cam_path): os.makedirs(tgt_cam_path)
        self.seg_id_dict = seg_id_dict
        
    def extract_data(self):
        for h5py_filename in tqdm.tqdm(sorted(os.listdir(self.ori_h5py_path))):
            filename = h5py_filename.split('.')[0]
            with h5py.File(os.path.join(self.ori_h5py_path, h5py_filename), 'r') as h5py_f:
                seg_keyframe = h5py_f['seg_keyframe'][:]
                #assert is_consecutive(seg_keyframe[:, 1].tolist()) and seg_keyframe.shape[0] == 9
                
                frame_ranges = []
                for ori_seg_key in self.seg_id_dict.keys():
                    if ori_seg_key == 0:
                        frame_range = range(0, seg_keyframe[0][0])
                    else:
                        frame_range = None
                        for cnt, ele in enumerate(seg_keyframe):
                            if ele[1] == ori_seg_key:
                                if cnt < seg_keyframe.shape[0] - 1:
                                    frame_range = range(ele[0], seg_keyframe[cnt + 1][0])
                                else:
                                    frame_range = range(ele[0], h5py_f['action'].shape[0])
                                break
                        if frame_range == None: raise Exception(f"The seg id {ori_seg_key} is not found in {h5py_filename}!")
                    frame_ranges.append(frame_range)
                self.cp_h5py_data(h5py_f, h5py_filename, frame_ranges)

            for cam_name in self.cam_names:
                self.cp_video_data(cam_name, filename, frame_ranges)
        print('Done!')

    def cp_h5py_data(self, h5py_f, h5py_filename, frame_ranges):
        with h5py.File(os.path.join(self.tgt_h5py_path, h5py_filename), 'w') as tgt_h5py_f:
            action = []
            for frame_range in frame_ranges:
                action.append(h5py_f['action'][frame_range])
            action = np.concatenate(action, axis = 0)
            tgt_h5py_f['action'] = action
            
            new_frame_range = []
            ele_count = 0
            for cnt, frame_range in enumerate(frame_ranges):
                new_frame_range.append((ele_count, cnt))
                ele_count += len(frame_range)
            new_frame_range = np.array(new_frame_range)
            tgt_h5py_f['seg_keyframe'] = new_frame_range
            
            tgt_h5py_observations = tgt_h5py_f.create_group('observations')
            for obs_key in h5py_f['observations'].keys():
                obs = []
                for frame_range in frame_ranges:
                    obs.append(h5py_f['observations'][obs_key][frame_range])
                obs = np.concatenate(obs, axis = 0)
                tgt_h5py_observations[obs_key] = obs

    def cp_video_data(self, cam_name, filename, frame_ranges):
        src_video_path = os.path.join(self.ori_root, cam_name, filename + '.mp4')
        tgt_video_path = os.path.join(self.tgt_root, cam_name, filename + '.mp4')
        if os.path.exists(tgt_video_path): os.remove(tgt_video_path)

        read_cap = cv2.VideoCapture(src_video_path)
        write_cap = cv2.VideoWriter(tgt_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (640, 480))

        for frame_range in frame_ranges:
            for frame_id in frame_range:
                read_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ret, frame = read_cap.read()
                write_cap.write(frame)

        read_cap.release()
        write_cap.release()

def is_consecutive(lst):
    #sorted_lst = sorted(lst)
    return all(lst[i] + 1 == lst[i + 1] for i in range(len(lst) - 1))

if __name__ == '__main__':
    ori_root =  '/home/cvte/twilight/data/aloha_beverage'
    seg_id_dict = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
    assert is_consecutive(list(seg_id_dict.keys()))
    tgt_root = '/home/cvte/twilight/data/aloha_pourblueplate'

    de = DataExtractor(ori_root, tgt_root, seg_id_dict)
    de.extract_data()