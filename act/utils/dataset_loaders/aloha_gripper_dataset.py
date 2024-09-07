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
import math
import matplotlib.pyplot as plt

from utils import comm
from utils import samplers

class AlohaGripperDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, transforms, norm_stats, ids_map_dict, indices, is_train):
        super(AlohaGripperDataset).__init__()
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
            obs_sample_interval = obs_sample_interval * self.cfg['DATA']['ACTION_SAMPLE_INTERVAL']
            if start_ts >= (past_obs_len - 1) * obs_sample_interval:
                effort_obs = root['/observations/effort_obs'][start_ts - (past_obs_len - 1) * obs_sample_interval : start_ts + 1 : obs_sample_interval]
                qpos_obs = root['/observations/qpos_obs'][start_ts - (past_obs_len - 1) * obs_sample_interval : start_ts + 1 : obs_sample_interval]
                qvel_obs = root['/observations/qvel_obs'][start_ts - (past_obs_len - 1) * obs_sample_interval : start_ts + 1 : obs_sample_interval]
                observation_len = self.cfg['DATA']['PAST_OBSERVATION_LEN']
            else:
                effort_obs = np.zeros((past_obs_len, root['/observations/effort_obs'][:].shape[1]), np.float32)
                qpos_obs = np.zeros((past_obs_len, root['/observations/qpos_obs'][:].shape[1]), np.float32)
                qvel_obs = np.zeros((past_obs_len, root['/observations/qvel_obs'][:].shape[1]), np.float32)
                valid_past_num = start_ts // obs_sample_interval
                st = start_ts - valid_past_num * obs_sample_interval
                effort_obs[-valid_past_num - 1:] = root['/observations/effort_obs'][st : start_ts + 1 : obs_sample_interval]
                qpos_obs[-valid_past_num - 1:] = root['/observations/qpos_obs'][st : start_ts + 1 : obs_sample_interval]
                qvel_obs[-valid_past_num - 1:] = root['/observations/qvel_obs'][st : start_ts + 1 : obs_sample_interval]
                observation_len = valid_past_num + 1
            observation_is_pad = np.zeros(self.cfg['DATA']['PAST_OBSERVATION_LEN'])
            observation_is_pad[:-observation_len] = 1   # Invalid observation
            
            image_dict = dict()
            for cam_name in self.camera_names:
                video_path = os.path.join(self.dataset_dir, cam_name, hdf5_file_name[:-4] + 'mp4')
                image_dict[cam_name] = self.load_video_frame(video_path, start_ts)
            
            past_action_len, past_action_interval = self.cfg['DATA']['PAST_ACTION_LEN'], self.cfg['DATA']['PAST_ACTION_SAMPLE_INTERVAL']
            past_action_interval = past_action_interval * self.cfg['DATA']['ACTION_SAMPLE_INTERVAL']
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
            past_action_is_pad[:-past_action_len] = 1   # Invalid past action
            
            chunk_size = self.cfg['POLICY']['CHUNK_SIZE']
            action_sample_interval = self.cfg['DATA']['ACTION_SAMPLE_INTERVAL']
            if root['/action'].shape[0] >= start_ts + chunk_size * action_sample_interval + 1:
                action = root['/action'][start_ts + 1 : start_ts + chunk_size * action_sample_interval + 1 : action_sample_interval]
                action_len = self.cfg['POLICY']['CHUNK_SIZE']
            else:
                action = root['/action'][start_ts + 1 :: action_sample_interval]
                action_len = math.ceil((episode_len - start_ts - 1) / action_sample_interval)
            padded_action = np.zeros((self.cfg['POLICY']['CHUNK_SIZE'], action.shape[1]), dtype=np.float32)
            padded_action[:action_len] = action
            action_is_pad = np.zeros(self.cfg['POLICY']['CHUNK_SIZE'])
            action_is_pad[action_len:] = 1

            '''action_array = padded_action[:, :7]
            plt.figure(figsize=(10, 8))
            for i in range(action_array.shape[1]):
                plt.subplot(action_array.shape[1], 1, i + 1)
                plt.scatter(range(action_array.shape[0]), action_array[:, i], label=f'Dimension {i+1}')
                plt.legend(loc='upper right')
                plt.title(f'Joint {i+1}')
            plt.tight_layout()
            plt.savefig('vis.png')
            pdb.set_trace()'''

            #assert 'seg_keyframe' in root.keys(), f"seg_keyframe is missing in {hdf5_file_name}"
            if self.cfg['POLICY']['STATUS_PREDICT'] and 'seg_keyframe' in root.keys():
                seg_keyframe = root['/seg_keyframe'][:] # Left shape: (key_num, 2). The first number is frame id and the second one is status id.
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

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)   # Left shape: (n, h, w, 3)
        
        # construct observations
        image_data = torch.from_numpy(all_cam_images).float() / 255   # left shape: (n, h, w, c)
        effort_obs = torch.from_numpy(effort_obs).float()  # left shape: (obs_len, joint_dim)
        qpos_obs = torch.from_numpy(qpos_obs).float() # left shape: (obs_len, joint_dim)
        qvel_obs = torch.from_numpy(qvel_obs).float() # left shape: (obs_len, joint_dim)
        past_action = torch.from_numpy(past_action).float()   # left shape: (past_action_len, action_dim)
        action_data = torch.from_numpy(padded_action).float()   # left shape: (chunk_size, action_dim)
        observation_is_pad = torch.from_numpy(observation_is_pad).bool()    # left shape: (obs_len,)
        past_action_is_pad = torch.from_numpy(past_action_is_pad).bool()    # left shape: (past_action_len,)
        action_is_pad = torch.from_numpy(action_is_pad).bool()    # left shape: (chunk_size,)
        
        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)
        
        past_action = (past_action - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        effort_obs = (effort_obs - self.norm_stats['observations/effort_obs_mean']) / self.norm_stats['observations/effort_obs_std']
        qpos_obs = (qpos_obs - self.norm_stats['observations/qpos_obs_mean']) / self.norm_stats['observations/qpos_obs_std']
        qvel_obs = (qvel_obs - self.norm_stats['observations/qvel_obs_mean']) / self.norm_stats['observations/qvel_obs_std']

        image_data = self.transforms(image_data)
        
        return image_data, past_action.float(), action_data.float(), effort_obs.float(), qpos_obs.float(), qvel_obs.float(), observation_is_pad, past_action_is_pad, action_is_pad, status
    
class GripperImgNormalize():
    def __init__(self, mean, std, to_bgr=False):
        self.mean = mean
        self.std = std
        self.to_bgr = to_bgr

    def __call__(self, image_data,):
        image_data = transforms.functional.normalize(image_data, mean=self.mean, std=self.std)
        if self.to_bgr:
            image_data = image_data[:, [2, 1, 0], :, :]
        
        return image_data
    
class GripperImageCompose():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image_data,):
        for t in self.transforms:
            image_data = t(image_data)
        return image_data
    
def build_AlohaGripperTransforms(cfg, is_train=True):
    resize_transform = transforms.Resize((cfg['DATA']['IMG_RESIZE_SHAPE'][1], cfg['DATA']['IMG_RESIZE_SHAPE'][0]))
    normalize_transform = GripperImgNormalize(
        mean=cfg['DATA']['IMG_NORM_MEAN'], std=cfg['DATA']['IMG_NORM_STD'], to_bgr=False,
    )

    transform = GripperImageCompose(
        [
            resize_transform,
            normalize_transform,
        ]
    )
    return transform