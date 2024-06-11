import numpy as np
import torch
import os
import random
import logging
import pdb
import h5py
from torch.utils.data import TensorDataset, DataLoader
import IPython
e = IPython.embed

from utils import comm
from utils import samplers
from utils.transforms.build import build_transforms

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, transforms, norm_stats, ids_map_dict, indices, is_train):
        super(EpisodicDataset).__init__()
        self.cfg = cfg
        self.transforms = transforms
        self.norm_stats = norm_stats
        self.is_sim = None
        self.ids_map_dict = ids_map_dict
        self.indices = indices
        self.is_train = is_train
        self.dataset_dir = self.cfg['DATASET_DIR']
        self.camera_names = self.cfg['DATA']['CAMERA_NAMES']
        self.is_sim = 'sim_' in self.cfg['TASK_NAME']
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

def get_norm_stats(dataset_dir, norm_keys):
    norm_data_dict = {key: [] for key in norm_keys}
    mean_std_dict = {}
    for key in norm_keys:
        mean_std_dict[key + '_mean'] = []
        mean_std_dict[key + '_std'] = []

    data_file_list = get_hdf5_list(dataset_dir)

    for data_file in data_file_list:
        dataset_path = os.path.join(dataset_dir, data_file)
        with h5py.File(dataset_path, 'r') as root:
            for norm_key in norm_keys:
                norm_data_dict[norm_key].append(torch.from_numpy(root[norm_key][()]))

    for norm_key in norm_keys:
        norm_data_dict[norm_key] = torch.stack(norm_data_dict[norm_key])
        mean_std_dict[norm_key + '_mean'] = norm_data_dict[norm_key].mean(dim=[0, 1])
        mean_std_dict[norm_key + '_std'] = norm_data_dict[norm_key].std(dim=[0, 1])
        mean_std_dict[norm_key + '_std'] = torch.clip(mean_std_dict[norm_key + '_std'], 1e-2, np.inf) # avoid the std to be too small.

    return mean_std_dict

def get_hdf5_list(path):
    hdf5_list = []
    for file_name in os.listdir(path):
        if '.hdf5' in file_name: 
            hdf5_list.append(file_name)

    if len(hdf5_list) == 0:
        raise Exception("No hdf5 file found in the path {}".format(path))
    
    return sorted(hdf5_list)

def load_data(cfg):
    dataset_dir = cfg['DATASET_DIR']
    norm_keys = cfg['DATA']['NORM_KEYS'] 
    is_debug = cfg['IS_DEBUG']
    data_eval_ratio = cfg['EVAL']['DATA_EVAL_RATIO']
    data_train_ratio = 1 - data_eval_ratio

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, norm_keys)
    ids_map_dict, max_idx = get_ids_map(dataset_dir)
    shuffled_indices = np.random.permutation(max_idx)
    train_indices = shuffled_indices[:int(data_train_ratio * max_idx)]
    val_indices = shuffled_indices[int(data_train_ratio * max_idx):]
    
    # construct dataset and dataloader
    train_transforms = build_transforms(cfg, is_train = True)
    train_dataset = EpisodicDataset(cfg, transforms = train_transforms, norm_stats = norm_stats, ids_map_dict = ids_map_dict, indices = train_indices, is_train = True)
    val_transforms = build_transforms(cfg, is_train = False)
    val_dataset = EpisodicDataset(cfg, transforms = val_transforms, norm_stats = norm_stats, ids_map_dict = ids_map_dict, indices = val_indices, is_train = False)

    if is_debug:
        num_workers = 0
    else:
        num_workers = 1
    
    train_sample_per_gpu = cfg['TRAIN']['BATCH_SIZE'] // comm.get_world_size()
    val_sample_per_gpu = cfg['EVAL']['BATCH_SIZE'] // comm.get_world_size()
    train_sampler = samplers.TrainingSampler(len(train_dataset))
    val_sampler = samplers.InferenceSampler(len(val_dataset))
    train_batch_sampler = torch.utils.data.sampler.BatchSampler(train_sampler, train_sample_per_gpu, drop_last=True)
    val_batch_sampler = torch.utils.data.sampler.BatchSampler(val_sampler, val_sample_per_gpu, drop_last=True)

    train_dataloader = DataLoader(train_dataset, num_workers=num_workers, batch_sampler=train_batch_sampler)
    val_dataloader = DataLoader(val_dataset, num_workers=num_workers, batch_sampler=val_batch_sampler)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim

def get_ids_map(dataset_dir):
    data_file_list = get_hdf5_list(dataset_dir)
    idx_start = 0
    ids_map = {}
    for data_file in data_file_list:
        with h5py.File(os.path.join(dataset_dir, data_file), 'r') as root:
            episode_len = root['action'].shape[0]
            ids_map[data_file] = (idx_start, idx_start + episode_len - 1)
            idx_start += episode_len
    return ids_map, idx_start


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

