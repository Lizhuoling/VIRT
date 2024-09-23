import numpy as np
import torch
import os
import random
import cv2
import logging
import pdb
import h5py
import math
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import functional as F
from torchvision import transforms
import IPython
e = IPython.embed

from utils import comm
from utils import samplers

class ACTDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, transforms, norm_stats, ids_map_dict, indices, is_train):
        super(ACTDataset).__init__()
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

            # get observation at start_ts only
            if 'end_observation' in root['observations'].keys():
                qpos = root['/observations/end_observation'][start_ts]
            elif 'qpos_obs' in root['observations'].keys():
                qpos = root['/observations/qpos_obs'][start_ts]
            else:
                raise Exception("Not supported qpos format yet.")
            
            image_dict = dict()
            for cam_name in self.camera_names:
                video_path = os.path.join(self.dataset_dir, cam_name, hdf5_file_name[:-4] + 'mp4')
                image_dict[cam_name] = self.load_video_frame(video_path, start_ts)
            
            action_sample_interval = self.cfg['DATA']['ACTION_SAMPLE_INTERVAL']
            if root['/action'].shape[0] >= start_ts + self.cfg['POLICY']['CHUNK_SIZE'] * action_sample_interval + 1:
                action = root['/action'][start_ts + 1 : start_ts + self.cfg['POLICY']['CHUNK_SIZE'] * action_sample_interval + 1 : action_sample_interval]
                action_len = self.cfg['POLICY']['CHUNK_SIZE']
            else:
                action = root['/action'][start_ts + 1 :: action_sample_interval]
                action_len = math.ceil((episode_len - start_ts - 1) / action_sample_interval)
            padded_action = np.zeros((self.cfg['POLICY']['CHUNK_SIZE'], action.shape[1]), dtype=np.float32)
            padded_action[:action_len] = action
            is_pad = np.zeros(self.cfg['POLICY']['CHUNK_SIZE'])
            is_pad[action_len:] = 1

            if self.cfg['TASK_NAME'] == 'isaac_singlebox':
                task_instruction = 'red'
            elif self.cfg['TASK_NAME'] in ['isaac_singlebox', 'isaac_singlecolorbox', 'isaac_multicolorbox']:
                task_instruction = np.array(root['/task_instruction']).item().decode('utf-8')
            else:
                task_instruction = 'none'

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images).float() / 255   # left shape: (n, h, w, c)
        qpos_data = torch.from_numpy(qpos).float()  # left shape: (pos_len, )
        action_data = torch.from_numpy(padded_action).float()   # left shape: (chunk_size, action_len)
        is_pad = torch.from_numpy(is_pad).bool()    # left shape: (chunk_size,)
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        action_data = (action_data - self.norm_stats["action_mean"].float()) / self.norm_stats["action_std"].float()
        if 'observations/end_observation_mean' in self.norm_stats.keys():
            qpos_data = (qpos_data - self.norm_stats['observations/end_observation_mean'].float()) / self.norm_stats['observations/end_observation_std'].float()
        elif 'observations/qpos_obs_mean' in self.norm_stats.keys():
            qpos_data = (qpos_data - self.norm_stats['observations/qpos_obs_mean'].float()) / self.norm_stats['observations/qpos_obs_std'].float()
        else:
            raise NotImplementedError
        
        image_data = self.transforms(image_data)
        
        return image_data, qpos_data, action_data, is_pad, task_instruction
    
class ACTCompose():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image_data,):
        for t in self.transforms:
            image_data = t(image_data)
        return image_data

class ACTToTensor():
    def __call__(self, image_data):
        return F.to_tensor(image_data)

class ACTNormalize():
    def __init__(self, mean, std, to_bgr=False):
        self.mean = mean
        self.std = std
        self.to_bgr = to_bgr

    def __call__(self, image_data):
        image_data = F.normalize(image_data, mean=self.mean, std=self.std)

        if self.to_bgr:
            image_data = image_data[:, [2, 1, 0], :, :]
        
        return image_data
    
def build_ACTTransforms(cfg, is_train=True):
    resize_transform = transforms.Resize((cfg['DATA']['IMG_RESIZE_SHAPE'][1], cfg['DATA']['IMG_RESIZE_SHAPE'][0]))
    normalize_transform = ACTNormalize(
        mean=cfg['DATA']['IMG_NORM_MEAN'], std=cfg['DATA']['IMG_NORM_STD'], to_bgr=False,
    )

    transform = ACTCompose(
        [
            resize_transform,
            normalize_transform,
        ]
    )
    return transform