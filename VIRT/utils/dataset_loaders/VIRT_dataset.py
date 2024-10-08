import numpy as np
import torch
import os
import random
import cv2
import logging
import pdb
import h5py
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
import IPython
e = IPython.embed

from utils import comm

class VIRTDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, transforms, norm_stats, ids_map_dict, indices, is_train):
        super(VIRTDataset).__init__()
        self.cfg = cfg
        self.transforms = transforms
        self.norm_stats = norm_stats
        self.ids_key_list = list(ids_map_dict.keys())
        self.ids_start_array = np.array([ele[0] for ele in ids_map_dict.values()])
        self.indices = indices
        self.is_train = is_train
        self.dataset_dir = self.cfg['DATASET_DIR']
        self.camera_names = self.cfg['DATA']['CAMERA_NAMES']
        self.logger = logging.getLogger("grasp")

        if comm.is_main_process():
            if self.is_train:
                self.logger.info("Training set sample number: {}.".format(len(self.indices)))
            else:
                self.logger.info("Validation set sample number: {}.".format(len(self.indices)))

    def __len__(self):
        return len(self.indices)
    
    def map_value_to_letter(self, index):
        range_id = np.searchsorted(self.ids_start_array, index, side = 'right') - 1
        hdf5_file = self.ids_key_list[range_id]
        return hdf5_file, index - self.ids_start_array[range_id]
    
    def load_video_frame(self, video_path, frame_id):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cap.release()
        return frame_rgb

    def __getitem__(self, index):
        if isinstance(index, torch.Tensor):
            index = index.item()
        indice_index  = self.indices[index]
        hdf5_file_name, hdf5_frame_id =  self.map_value_to_letter(indice_index)
        h5py_path = os.path.join(self.dataset_dir, 'h5py', hdf5_file_name)
        with h5py.File(h5py_path, 'r') as root:
            episode_len = root['/action'].shape[0]
            start_ts = hdf5_frame_id

            
            past_obs_len, obs_sample_interval = self.cfg['DATA']['PAST_OBSERVATION_LEN'], self.cfg['DATA']['OBSERVATION_SAMPLE_INTERVAL']
            end_observation = np.zeros((past_obs_len, root['/observations/end_observation'][:].shape[1]), np.float32)
            joint_observation = np.zeros((past_obs_len, root['/observations/joint_observation'][:].shape[1]), np.float32)
            if start_ts >= (past_obs_len - 1) * obs_sample_interval:
                end_observation = root['/observations/end_observation'][start_ts - (past_obs_len - 1) * obs_sample_interval : start_ts + 1 : obs_sample_interval]
                joint_observation = root['/observations/joint_observation'][start_ts - (past_obs_len - 1) * obs_sample_interval : start_ts + 1 : obs_sample_interval]
                observation_len = self.cfg['DATA']['PAST_OBSERVATION_LEN']
            else:
                valid_past_num = start_ts // obs_sample_interval
                st = start_ts - valid_past_num * obs_sample_interval
                end_observation[-valid_past_num - 1:] = root['/observations/end_observation'][st : start_ts + 1 : obs_sample_interval]
                joint_observation[-valid_past_num - 1:] = root['/observations/joint_observation'][st : start_ts + 1 : obs_sample_interval]
                observation_len = valid_past_num + 1
            observation_is_pad = np.zeros(self.cfg['DATA']['PAST_OBSERVATION_LEN'])
            observation_is_pad[:-observation_len] = 1   
            
            image_dict = dict()
            for cam_name in self.camera_names:
                video_path = os.path.join(self.dataset_dir, cam_name, hdf5_file_name[:-4] + 'mp4')
                image_dict[cam_name] = self.load_video_frame(video_path, start_ts)
            
            past_action_len, past_action_interval = self.cfg['DATA']['PAST_ACTION_LEN'], self.cfg['DATA']['PAST_ACTION_SAMPLE_INTERVAL']
            past_action = np.zeros((past_action_len, root['/action'][:].shape[1]), np.float32)
            if start_ts >= (past_action_len - 1) * past_action_interval:
                past_action = root['/action'][start_ts - (past_action_len - 1) * past_action_interval : start_ts + 1 : past_action_interval]
                past_action_len = self.cfg['DATA']['PAST_ACTION_LEN']
            else:
                valid_past_num = start_ts // past_action_interval
                st = start_ts - valid_past_num * past_action_interval
                past_action[-valid_past_num - 1:] = root['/action'][st : start_ts + 1 : past_action_interval]
                past_action_len = valid_past_num + 1
            past_action_is_pad = np.zeros(self.cfg['DATA']['PAST_ACTION_LEN'])
            past_action_is_pad[:-past_action_len] = 1   
            
            if root['/action'].shape[0] >= start_ts + self.cfg['POLICY']['CHUNK_SIZE'] + 1:
                action = root['/action'][start_ts + 1 : start_ts + self.cfg['POLICY']['CHUNK_SIZE'] + 1]
                action_len = self.cfg['POLICY']['CHUNK_SIZE']
            else:
                action = root['/action'][start_ts + 1:]
                action_len = episode_len - start_ts - 1
            padded_action = np.zeros((self.cfg['POLICY']['CHUNK_SIZE'], action.shape[1]), dtype=np.float32)
            padded_action[:action_len] = action
            action_is_pad = np.zeros(self.cfg['POLICY']['CHUNK_SIZE'])
            action_is_pad[action_len:] = 1

            if self.cfg['TASK_NAME'] == 'isaac_singlebox':
                task_instruction = 'red'
            else:
                task_instruction = np.array(root['/task_instruction']).item().decode('utf-8')

            if self.cfg['POLICY']['STATUS_PREDICT'] and 'seg_keyframe' in root.keys():
                seg_keyframe = root['/seg_keyframe'][:] 
                if hdf5_frame_id < seg_keyframe[0][0]: 
                    status = 0
                elif hdf5_frame_id >= seg_keyframe[0][0] and hdf5_frame_id < seg_keyframe[-1][0]:
                    for i in range(seg_keyframe.shape[0] - 1):
                        if hdf5_frame_id >= seg_keyframe[i][0] and hdf5_frame_id < seg_keyframe[i + 1][0]:
                            status = seg_keyframe[i][1]
                            break
                else:
                    status = seg_keyframe[-1][1]
            else:
                status = 0

        
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)   
        
        
        image_data = torch.from_numpy(all_cam_images).float() / 255   
        end_observation = torch.from_numpy(end_observation).float()  
        joint_observation = torch.from_numpy(joint_observation).float() 
        past_action = torch.from_numpy(past_action).float()   
        action_data = torch.from_numpy(padded_action).float()   
        observation_is_pad = torch.from_numpy(observation_is_pad).bool()    
        past_action_is_pad = torch.from_numpy(past_action_is_pad).bool()    
        action_is_pad = torch.from_numpy(action_is_pad).bool()    
        
        
        image_data = torch.einsum('k h w c -> k c h w', image_data)
        
        action_mean_key, action_std_key = self.retrieve_key(self.norm_stats.keys(), "action_mean"), self.retrieve_key(self.norm_stats.keys(), "action_std")
        past_action = (past_action - self.norm_stats[action_mean_key]) / self.norm_stats[action_std_key]
        action_data = (action_data - self.norm_stats[action_mean_key]) / self.norm_stats[action_std_key]
        
        endobs_mean_key, endobs_std_key = self.retrieve_key(self.norm_stats.keys(), "observations/end_observation_mean"), self.retrieve_key(self.norm_stats.keys(), "observations/end_observation_std")
        end_observation = (end_observation - self.norm_stats[endobs_mean_key]) / self.norm_stats[endobs_std_key]

        jointobs_mean_key, jointobs_std_key = self.retrieve_key(self.norm_stats.keys(), "observations/joint_observation_mean"), self.retrieve_key(self.norm_stats.keys(), "observations/joint_observation_std")
        joint_observation = (joint_observation - self.norm_stats[jointobs_mean_key]) / self.norm_stats[jointobs_std_key]

        image_data = self.transforms(image_data)

        return image_data, past_action, action_data, end_observation, joint_observation, observation_is_pad, past_action_is_pad, action_is_pad, task_instruction, status

    def retrieve_key(self, key_list, keyword):
        for key in key_list:
            if keyword == key: return key

        raise Exception("Keyword {} is not found in the key list {}".format(keyword, key_list))
    
class ImgNormalize():
    def __init__(self, mean, std, to_bgr=False):
        self.mean = mean
        self.std = std
        self.to_bgr = to_bgr

    def __call__(self, image_data,):
        image_data = transforms.functional.normalize(image_data, mean=self.mean, std=self.std)
        if self.to_bgr:
            image_data = image_data[:, [2, 1, 0], :, :]
        
        return image_data
    
class ImageCompose():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image_data,):
        for t in self.transforms:
            image_data = t(image_data)
        return image_data
    
def build_VIRTTransforms(cfg, is_train=True):
    resize_transform = transforms.Resize((cfg['DATA']['IMG_RESIZE_SHAPE'][1], cfg['DATA']['IMG_RESIZE_SHAPE'][0]))
    normalize_transform = ImgNormalize(
        mean=cfg['DATA']['IMG_NORM_MEAN'], std=cfg['DATA']['IMG_NORM_STD'], to_bgr=False,
    )

    transform = ImageCompose(
        [
            resize_transform,
            normalize_transform,
        ]
    )
    return transform