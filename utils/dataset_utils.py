import numpy as np
import torch
import os
import random
import pickle
import logging
import tqdm
import pdb
import h5py
from torch.utils.data import TensorDataset, DataLoader
import IPython
e = IPython.embed

from utils import comm
from utils import samplers
from utils.dataset_loaders.act_dataset import ACTDataset, build_ACTTransforms
from utils.dataset_loaders.isaac_gripper_dataset import IsaacGripperDataset, build_IsaacGripperTransforms
from utils.dataset_loaders.droid_pretrain_dataset import DroidPretrainDataset, build_DroidPretrainTransforms
from utils.dataset_loaders.aloha_gripper_dataset import AlohaGripperDataset, build_AlohaGripperTransforms

def get_norm_stats(dataset_dir, norm_keys, norm_max_len = -1):
    norm_data_dict = {key: [] for key in norm_keys}
    mean_std_dict = {}
    for key in norm_keys:
        mean_std_dict[key + '_mean'] = []
        mean_std_dict[key + '_std'] = []
    
    data_file_list = get_hdf5_list(os.path.join(dataset_dir, 'h5py'))
    if norm_max_len > 0 and len(data_file_list) > norm_max_len:
        data_file_list = data_file_list[:norm_max_len]

    for data_file in data_file_list:
        dataset_path = os.path.join(dataset_dir, 'h5py', data_file)
        with h5py.File(dataset_path, 'r') as root:
            for norm_key in norm_keys:
                norm_data_dict[norm_key].append(torch.from_numpy(root[norm_key][()]))

    for norm_key in norm_keys:
        norm_data_dict[norm_key] = torch.cat(norm_data_dict[norm_key], axis = 0)
        mean_std_dict[norm_key + '_mean'] = norm_data_dict[norm_key].mean(dim=0)
        mean_std_dict[norm_key + '_std'] = norm_data_dict[norm_key].std(dim=0)
        mean_std_dict[norm_key + '_std'] = torch.clip(mean_std_dict[norm_key + '_std'], 1e-2, np.inf) # avoid the std to be too small.
    
    return mean_std_dict

def get_hdf5_list(path):
    hdf5_list = []
    for file_name in os.listdir(path):
        if file_name.endswith('.hdf5'): 
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
    norm_max_len = cfg['DATA']['NORM_MAX_LEN']

    if cfg["DATA"]["LOAD_INDICE_PATH"] == "":
        # obtain normalization stats for qpos and action
        norm_stats = get_norm_stats(dataset_dir, norm_keys, norm_max_len)
        ids_map_dict, max_idx = get_ids_map(dataset_dir)
        if cfg['TRAIN']['DATA_SAMPLE_MODE'] == 'random':
            shuffled_indices = np.random.permutation(max_idx)
        elif cfg['TRAIN']['DATA_SAMPLE_MODE'] == 'sequence':
            shuffled_indices = get_sequence_indices(ids_map_dict = ids_map_dict, chunk_size = cfg['POLICY']['CHUNK_SIZE'])
            random.shuffle(shuffled_indices)
        '''indice_data = dict(norm_stats = norm_stats, ids_map_dict = ids_map_dict, max_idx = max_idx, shuffled_indices = shuffled_indices)
        with open("/home/cvte/twilight/home/data/droid_h5py/droid_pretrain_indice.pkl", 'wb') as file:
            pickle.dump(indice_data, file)'''
    else:
        with open(cfg["DATA"]["LOAD_INDICE_PATH"], 'rb') as file:
            indice_data = pickle.load(file)
        norm_stats, ids_map_dict, max_idx, shuffled_indices = indice_data['norm_stats'], indice_data['ids_map_dict'], indice_data['max_idx'], indice_data['shuffled_indices']
        
    if data_eval_ratio > 0: 
        train_indices = shuffled_indices[:int(data_train_ratio * max_idx)]
        val_indices = shuffled_indices[int(data_train_ratio * max_idx):]
    else:
        train_indices = shuffled_indices
    
    # construct dataset and dataloader
    if cfg['TASK_NAME'] == 'own_gripper':
        train_transforms = build_ACTTransforms(cfg, is_train = True)
        train_dataset = ACTDataset(cfg, transforms = train_transforms, norm_stats = norm_stats, ids_map_dict = ids_map_dict, indices = train_indices, is_train = True)
        if data_eval_ratio > 0:
            val_transforms = build_ACTTransforms(cfg, is_train = False)
            val_dataset = ACTDataset(cfg, transforms = val_transforms, norm_stats = norm_stats, ids_map_dict = ids_map_dict, indices = val_indices, is_train = False)
    elif cfg['TASK_NAME'] in ['isaac_multicolorbox', 'isaac_singlebox', 'isaac_singlecolorbox', 'isaac_twoboxred', 'isaac_fiveboxred']:
        train_transforms = build_IsaacGripperTransforms(cfg, is_train = True)
        train_dataset = IsaacGripperDataset(cfg, transforms = train_transforms, norm_stats = norm_stats, ids_map_dict = ids_map_dict, indices = train_indices, is_train = True)
        if data_eval_ratio > 0:
            val_transforms = build_IsaacGripperTransforms(cfg, is_train = False)
            val_dataset = IsaacGripperDataset(cfg, transforms = val_transforms, norm_stats = norm_stats, ids_map_dict = ids_map_dict, indices = val_indices, is_train = False)
    elif cfg['TASK_NAME'] == 'droid_pretrain':
        train_transforms = build_DroidPretrainTransforms(cfg, is_train = True)
        train_dataset = DroidPretrainDataset(cfg, transforms = train_transforms, norm_stats = norm_stats, ids_map_dict = ids_map_dict, indices = train_indices, is_train = True)
    elif cfg['TASK_NAME'] in ['aloha_singleobjgrasp',]:
        train_transforms = build_AlohaGripperTransforms(cfg, is_train = True)
        train_dataset = AlohaGripperDataset(cfg, transforms = train_transforms, norm_stats = norm_stats, ids_map_dict = ids_map_dict, indices = train_indices, is_train = True)

    if is_debug:
        num_workers = 0
    else:
        num_workers = 3
    
    train_sample_per_gpu = cfg['TRAIN']['BATCH_SIZE'] // comm.get_world_size()
    train_sampler = samplers.TrainingSampler(len(train_dataset))
    train_batch_sampler = torch.utils.data.sampler.BatchSampler(train_sampler, train_sample_per_gpu, drop_last=True)
    train_dataloader = DataLoader(train_dataset, num_workers=num_workers, batch_sampler=train_batch_sampler)
    
    if data_eval_ratio > 0: 
        val_sample_per_gpu = cfg['EVAL']['BATCH_SIZE'] // comm.get_world_size()
        val_sampler = samplers.InferenceSampler(len(val_dataset))
        val_batch_sampler = torch.utils.data.sampler.BatchSampler(val_sampler, val_sample_per_gpu, drop_last=True)
        val_dataloader = DataLoader(val_dataset, num_workers=num_workers, batch_sampler=val_batch_sampler)
        return train_dataloader, val_dataloader, norm_stats
    else:
        return train_dataloader, None, norm_stats

def get_ids_map(dataset_dir):
    data_file_list = get_hdf5_list(os.path.join(dataset_dir, 'h5py'))
    idx_start = 0
    ids_map = {}
    for data_file in tqdm.tqdm(data_file_list):
        with h5py.File(os.path.join(dataset_dir, 'h5py', data_file), 'r') as root:
            episode_len = root['action'].shape[0]
            ids_map[data_file] = (idx_start, idx_start + episode_len - 1)
            idx_start += episode_len
    return ids_map, idx_start

def get_sequence_indices(ids_map_dict, chunk_size):
    indices = []
    for key, value in ids_map_dict.items():
        start, end = value
        indices += [ele for ele in range(start, end, chunk_size)]
    return indices

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

