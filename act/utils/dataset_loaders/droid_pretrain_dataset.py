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
from utils import samplers

class DroidPretrainDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, transforms, norm_stats, ids_map_dict, indices, is_train):
        super(DroidPretrainDataset).__init__()
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
            past_obs_len, obs_sample_interval = self.cfg['DATA']['PAST_OBSERVATION_LEN'], self.cfg['DATA']['OBSERVATION_SAMPLE_INTERVAL']
            whole_cartesian_position = root['observation/cartesian_position'][:]    # Left shape: (T, 6)
            whole_gripper_position = root['observation/gripper_position'][:]    # Left shape: (T, 1)
            whole_end_position = np.concatenate((whole_cartesian_position, whole_gripper_position), axis = 1)   # Left shape: (T, 7)
            whole_joint_position = root['observation/joint_position'][:]    # Left shape: (T, 7)

            end_observation = np.zeros((past_obs_len, whole_end_position.shape[1]), np.float32)
            joint_observation = np.zeros((past_obs_len, whole_joint_position.shape[1]), np.float32)
            if start_ts >= (past_obs_len - 1) * obs_sample_interval:
                end_observation = whole_end_position[start_ts - (past_obs_len - 1) * obs_sample_interval : start_ts + 1 : obs_sample_interval]
                joint_observation = whole_joint_position[start_ts - (past_obs_len - 1) * obs_sample_interval : start_ts + 1 : obs_sample_interval]
                observation_len = self.cfg['DATA']['PAST_OBSERVATION_LEN']
            else:
                valid_past_num = start_ts // obs_sample_interval
                st = start_ts - valid_past_num * obs_sample_interval
                end_observation[-valid_past_num - 1:] = whole_end_position[st : start_ts + 1 : obs_sample_interval]
                joint_observation[-valid_past_num - 1:] = whole_joint_position[st : start_ts + 1 : obs_sample_interval]
                observation_len = valid_past_num + 1
            observation_is_pad = np.zeros(self.cfg['DATA']['PAST_OBSERVATION_LEN'])
            observation_is_pad[:-observation_len] = 1   # Invalid observation
            
            # Observation images
            image_dict = dict()
            for cam_name in self.camera_names:
                video_path = os.path.join(self.dataset_dir, cam_name, hdf5_file_name[:-4] + 'mp4')
                image_dict[cam_name] = self.load_video_frame(video_path, start_ts)  # rgb image

            # Future goal images
            goal_image_dict = dict()
            if self.cfg['DATA']['PRETRAIN_VISION_INSTRUCTION'] == "grasp_keyframe":
                info_path = os.path.join(self.dataset_dir, 'info', hdf5_file_name)
                goal_frame_idx = episode_len - 1    # Set the last frame as the initial default value.
                cam_map_dict = {'exterior_image_1_left': 'exterior1_grasp_frames', 'exterior_image_2_left': 'exterior2_grasp_frames', 'wrist_image_left': 'wrist_grasp_frames'}
                if os.path.exists(info_path):
                    with h5py.File(info_path, 'r') as info_root:
                        grasp_moments = info_root['/grasp_moments'][:]
                        grasp_moment_id = None
                        for grasp_idx, grasp_moment in enumerate(grasp_moments):
                            if grasp_idx == 0 and hdf5_frame_id < grasp_moment:
                                goal_frame_idx = grasp_moment
                                grasp_moment_id = grasp_idx
                                break
                            elif grasp_idx > 0 and hdf5_frame_id < grasp_moment and hdf5_frame_id >= grasp_moments[grasp_idx - 1]:
                                goal_frame_idx = grasp_moment
                                grasp_moment_id = grasp_idx
                                break
                            else:
                                pass
                        if goal_frame_idx != episode_len - 1:
                            for cam_name in self.camera_names:
                                grasp_view_name = cam_map_dict[cam_name]
                                goal_image_dict[cam_name] = info_root[grasp_view_name][grasp_moment_id]    # rgb image
                # If the info file does not exist or goal_frame_idx == episode_len - 1, directly select the last frames of videos as the goal images.
                if goal_frame_idx == episode_len - 1:
                    for cam_name in self.camera_names:
                        video_path = os.path.join(self.dataset_dir, cam_name, hdf5_file_name[:-4] + 'mp4')
                        goal_image_dict[cam_name] = self.load_video_frame(video_path, goal_frame_idx)  # rgb image
            elif self.cfg['DATA']['PRETRAIN_VISION_INSTRUCTION'] == 'future_frame':
                if start_ts + self.cfg['POLICY']['CHUNK_SIZE'] - 1 < episode_len:
                    goal_frame_idx = start_ts + self.cfg['POLICY']['CHUNK_SIZE'] - 1
                else:
                    goal_frame_idx = episode_len - 1
                for cam_name in self.camera_names:
                    video_path = os.path.join(self.dataset_dir, cam_name, hdf5_file_name[:-4] + 'mp4')
                    goal_image_dict[cam_name] = self.load_video_frame(video_path, goal_frame_idx)  # rgb image
            else:
                raise NotImplementedError
            
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
            past_action_is_pad[:-past_action_len] = 1   # Invalid past action
            
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

            task_instruction = np.array(root['/task_instruction']).item().decode('utf-8')

        # camera images
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)   # Left shape: (n, h, w, 3)

        # goal_images
        goal_images = []
        for cam_name in self.camera_names:
            goal_images.append(goal_image_dict[cam_name])
        goal_images = np.stack(goal_images, axis=0)   # Left shape: (n, h, w, 3)
        
        # construct observations
        image_data = torch.from_numpy(all_cam_images).float() / 255   # left shape: (n, h, w, c)
        goal_image_data = torch.from_numpy(goal_images).float() / 255   # left shape: (n, h, w, c)
        end_observation = torch.from_numpy(end_observation).float()  # left shape: (obs_len, end_dim)
        joint_observation = torch.from_numpy(joint_observation).float() # left shape: (obs_len, joint_dim)
        past_action = torch.from_numpy(past_action).float()   # left shape: (past_action_len, action_dim)
        action_data = torch.from_numpy(padded_action).float()   # left shape: (chunk_size, action_dim)
        observation_is_pad = torch.from_numpy(observation_is_pad).bool()    # left shape: (obs_len,)
        past_action_is_pad = torch.from_numpy(past_action_is_pad).bool()    # left shape: (past_action_len,)
        action_is_pad = torch.from_numpy(action_is_pad).bool()    # left shape: (chunk_size,)
        
        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)
        goal_image_data = torch.einsum('k h w c -> k c h w', goal_image_data)
        
        past_action = (past_action - self.norm_stats['action_mean']) / self.norm_stats['action_std']
        action_data = (action_data - self.norm_stats['action_mean']) / self.norm_stats['action_std']
        
        cartesian_mean, cartesian_std = self.norm_stats['observation/cartesian_position_mean'], self.norm_stats['observation/cartesian_position_std']
        gripper_mean, gripper_std = self.norm_stats['observation/gripper_position_mean'], self.norm_stats['observation/gripper_position_std']
        obs_mean = torch.cat((cartesian_mean, gripper_mean), dim = 0)
        obs_std = torch.cat((cartesian_std, gripper_std), dim = 0)
        end_observation = (end_observation - obs_mean) / obs_std

        joint_observation = (joint_observation - self.norm_stats['observation/joint_position_mean']) / self.norm_stats['observation/joint_position_std']

        image_data = self.transforms(image_data)
        goal_image_data = self.transforms(goal_image_data)
        
        return image_data, goal_image_data, past_action.float(), action_data.float(), end_observation.float(), joint_observation.float(), observation_is_pad, past_action_is_pad, action_is_pad, task_instruction
    
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
    
def build_DroidPretrainTransforms(cfg, is_train=True):
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