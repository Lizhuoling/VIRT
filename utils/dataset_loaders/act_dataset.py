import numpy as np
import torch
import os
import random
import logging
import pdb
import h5py
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import functional as F
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
        self.ids_map_dict = ids_map_dict
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
        for hdf5_file, idx_range in self.ids_map_dict.items():
            if idx_range[0] <= index <= idx_range[1]:
                return hdf5_file, index - idx_range[0]
        raise Exception('Value out of range')

    def __getitem__(self, index):
        if isinstance(index, torch.Tensor):
            index = index.item()
        hdf5_file_name, hdf5_frame_id =  self.map_value_to_letter(index)
        dataset_path = os.path.join(self.dataset_dir, hdf5_file_name)
        with h5py.File(dataset_path, 'r') as root:
            episode_len = root['/action'].shape[0]
            start_ts = hdf5_frame_id

            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
            
            if root['/action'].shape[0] > start_ts + self.cfg['POLICY']['CHUNK_SIZE']:
                action = root['/action'][start_ts : start_ts + self.cfg['POLICY']['CHUNK_SIZE']]
                action_len = self.cfg['POLICY']['CHUNK_SIZE']
            else:
                action = root['/action'][start_ts:]
                action_len = episode_len - start_ts

        padded_action = np.zeros((self.cfg['POLICY']['CHUNK_SIZE'], action.shape[1]), dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(self.cfg['POLICY']['CHUNK_SIZE'])
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images).float()   # left shape: (n, h, w, c)
        qpos_data = torch.from_numpy(qpos).float()  # left shape: (pos_len, )
        action_data = torch.from_numpy(padded_action).float()   # left shape: (chunk_size, action_len)
        is_pad = torch.from_numpy(is_pad).bool()    # left shape: (chunk_size,)
        
        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)
        
        action_mean_key, action_std_key = self.retrieve_key(self.norm_stats.keys(), "action_mean"), self.retrieve_key(self.norm_stats.keys(), "action_std")
        action_data = (action_data - self.norm_stats[action_mean_key]) / self.norm_stats[action_std_key]

        qpos_mean_key, qpos_std_key = self.retrieve_key(self.norm_stats.keys(), "observations/qpos_mean"), self.retrieve_key(self.norm_stats.keys(), "observations/qpos_std")
        qpos_data = (qpos_data - self.norm_stats[qpos_mean_key]) / self.norm_stats[qpos_std_key]

        image_data, qpos_data, action_data, is_pad = self.transforms(image_data, qpos_data, action_data, is_pad)

        return image_data, qpos_data, action_data, is_pad

    def retrieve_key(self, key_list, keyword):
        for key in key_list:
            if keyword == key: return key

        raise Exception("Keyword {} is not found in the key list {}".format(keyword, key_list))
    
class ACTCompose():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image_data, qpos_data, action_data, is_pad):
        for t in self.transforms:
            image_data, qpos_data, action_data, is_pad = t(image_data, qpos_data, action_data, is_pad)
        return image_data, qpos_data, action_data, is_pad

class ACTToTensor():
    def __call__(self, image_data, qpos_data, action_data, is_pad):
        return F.to_tensor(image_data), F.to_tensor(qpos_data), F.to_tensor(action_data), F.to_tensor(action_data)

class ACTNormalize():
    def __init__(self, mean, std, to_bgr=False):
        self.mean = mean
        self.std = std
        self.to_bgr = to_bgr

    def __call__(self, image_data, qpos_data, action_data, is_pad):
        image_data = F.normalize(image_data, mean=self.mean, std=self.std)

        if self.to_bgr:
            image_data = image_data[:, [2, 1, 0], :, :]
        
        return image_data, qpos_data, action_data, is_pad
    
def build_ACTTransforms(cfg, is_train=True):
    normalize_transform = ACTNormalize(
        mean=cfg['DATA']['IMG_NORM_MEAN'], std=cfg['DATA']['IMG_NORM_STD'], to_bgr=False,
    )

    transform = ACTCompose(
        [
            normalize_transform,
        ]
    )
    return transform