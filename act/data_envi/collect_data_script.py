from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_rotate, quat_conjugate, quat_mul

import pdb
import math
import numpy as np
import torch
import random
import time
from pathlib import Path
import os
import copy
import h5py
import cv2
import json
from typing import Dict, Optional
from pyquaternion import Quaternion
from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt
from itertools import combinations

from gripper_singlebox import GripperSingleBox
from sim_singlebox_script import SingleBoxScript

from gripper_singlecolorbox import GripperSingleColorBox
from sim_singlecolorbox_script import SingleColorBoxScript

from gripper_multicolorbox import GripperMultiColorBox
from sim_multicolorbox_script import MultiColorBoxScript

from gripper_twoboxred import GripperTwoBoxRed
from sim_twoboxred_script import TwoBoxRedScript

from gripper_fiveboxred import GripperFiveBoxRed
from sim_fiveboxred_script import FiveBoxRedScript

class SaveDataManager():
    def __init__(self, data_root, start_idx = None):
        self.data_root = data_root
        assert os.path.exists(self.data_root), "The path {} does not exist.".fsormat(self.data_root)
        if start_idx != None:
            self.episode_index = start_idx
        elif os.path.exists(os.path.join(self.data_root, "h5py")):
            file_list = sorted(os.listdir(os.path.join(self.data_root, "h5py")))
            file_list = [ele for ele in file_list if ele.endswith('.hdf5')]
            max_id = -1
            for file_name in file_list:
                id = int(file_name[8:-5])
                if id > max_id:
                    max_id = id 
            self.episode_index = max_id + 1
        else:
            self.episode_index = 0

        self.h5py_path = os.path.join(self.data_root, 'h5py')
        self.exterior_camera1_path = os.path.join(self.data_root, 'exterior_camera1')
        self.exterior_camera2_path  = os.path.join(self.data_root, 'exterior_camera2')
        self.top_camera_path  = os.path.join(self.data_root, 'top_camera')
        self.wrist_camera_path =  os.path.join(self.data_root, 'wrist_camera')
        if not os.path.exists(self.h5py_path): os.makedirs(self.h5py_path)
        if not os.path.exists(self.exterior_camera1_path): os.makedirs(self.exterior_camera1_path)
        if not os.path.exists(self.exterior_camera2_path): os.makedirs(self.exterior_camera2_path)
        if not os.path.exists(self.top_camera_path): os.makedirs(self.top_camera_path)
        if not os.path.exists(self.wrist_camera_path): os.makedirs(self.wrist_camera_path)

    def set_episode_index(self, index):
        self.episode_index = index

    def save_data(self, action_list, observations_list, images_list, task_instruction, seed):
        assert len(action_list) == len(observations_list) == len(images_list) != 0
        img_height, img_width, _ = images_list[0]['top_bgr_image'].shape
        
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        exterior1_writter = cv2.VideoWriter(os.path.join(self.exterior_camera1_path, 'episode_{}.mp4'.format(self.episode_index)), fourcc, 20, (img_width, img_height))
        exterior2_writter = cv2.VideoWriter(os.path.join(self.exterior_camera2_path, 'episode_{}.mp4'.format(self.episode_index)), fourcc, 20, (img_width, img_height))
        top_writter = cv2.VideoWriter(os.path.join(self.top_camera_path, 'episode_{}.mp4'.format(self.episode_index)), fourcc, 20, (img_width, img_height))
        wrist_writter = cv2.VideoWriter(os.path.join(self.wrist_camera_path, 'episode_{}.mp4'.format(self.episode_index)), fourcc, 20, (img_width, img_height))

        h5f = h5py.File(os.path.join(self.h5py_path, 'episode_{}.hdf5'.format(self.episode_index)), 'w')
        h5f['task_instruction'] = task_instruction
        end_observation = np.array([ele['end_observation'] for ele in observations_list])
        joint_observation = np.array([ele['joint_observation'] for ele in observations_list])
        h5f_observations = h5f.create_group('observations')
        h5f_observations['end_observation'] = end_observation
        h5f_observations['joint_observation'] = joint_observation
        h5f['action'] = np.array(action_list)
        h5f['seed'] = seed
        h5f.close()

        camera_keys = ['top_bgr_image', 'side1_bgr_image', 'side2_bgr_image', 'hand_bgr_image']
        for image_dict in images_list:
            for key in camera_keys:
                if key == 'side1_bgr_image':
                    exterior1_writter.write(image_dict[key])
                elif key == 'side2_bgr_image':
                    exterior2_writter.write(image_dict[key])
                elif key == 'top_bgr_image':
                    top_writter.write(image_dict[key])
                elif key == 'hand_bgr_image':
                    wrist_writter.write(image_dict[key])

        exterior1_writter.release()
        exterior2_writter.release()
        top_writter.release()
        wrist_writter.release()

        self.episode_index += 1

def collect_data_main(task_name, save_data_path = "", total_data_num = 1):
    save_data_flag = False
    if save_data_path != "":
        save_data_flag = True
    if save_data_flag:
        save_data_manager = SaveDataManager(save_data_path)
    
    if task_name == 'isaac_singlebox':
        isaac_env = GripperSingleBox(num_envs = 1)
        script_manager = SingleBoxScript(isaac_env = isaac_env)
    elif task_name == 'isaac_singlecolorbox':
        isaac_env = GripperSingleColorBox(num_envs = 1)
        script_manager = SingleColorBoxScript(isaac_env = isaac_env)
    elif task_name == 'isaac_multicolorbox':
        isaac_env = GripperMultiColorBox(num_envs = 1)
        script_manager = MultiColorBoxScript(isaac_env = isaac_env)
    elif task_name == 'isaac_twoboxred':
        isaac_env = GripperTwoBoxRed(num_envs = 1)
        script_manager = TwoBoxRedScript(isaac_env = isaac_env)
    elif task_name == 'isaac_fiveboxred':
        isaac_env = GripperFiveBoxRed(num_envs = 1)
        script_manager = FiveBoxRedScript(isaac_env = isaac_env)

    last_time = time.time()
    simulation_step = 0
    last_step = 0
    ctrl_min_step = 2
    print('\033[93m' + "Task instruction: {}".format(isaac_env.task_instruction[0]) + '\033[0m')

    action_list, observations_list, images_list = [], [], []
    end_signal = False
    if save_data_flag:
        sample_cnt = save_data_manager.episode_index
    else:
        sample_cnt = 0
    print(f"Start collect sample {sample_cnt}")
    while True:
        cur_time = time.time()
        if simulation_step - last_step < ctrl_min_step:
            ctrl_interval_flag = False
        else:
            ctrl_interval_flag = True
            last_step = simulation_step
        isaac_env.update_simulator_before_ctrl()

        if ctrl_interval_flag:
            action, observations, end_signal = script_manager.update_pos_action()
            isaac_env.update_action_map()

        isaac_env.update_simulator_after_ctrl()
        images_envs = isaac_env.visualization_process()
        cv2.waitKey(1)
        assert len(images_envs) ==  1
        images = images_envs[0]
        
        # Save data.
        if save_data_flag and ctrl_interval_flag:
            action_list.append(action)
            observations_list.append(observations)
            images_list.append(images)

        if end_signal:
            isaac_env.clean_up()
            if save_data_flag:
                save_data_manager.save_data(action_list = action_list, observations_list = observations_list, images_list = images_list, task_instruction = isaac_env.task_instruction[0], seed = isaac_env.random_seed)
            isaac_env.init_simulate_env()
            script_manager.__init__(isaac_env = isaac_env)
            action_list, observations_list, images_list = [], [], []
            sample_cnt += 1
            print(f"Start collect sample {sample_cnt}")

        if sample_cnt >= total_data_num:
            break
        simulation_step += 1
            
    isaac_env.clean_up()
    cv2.destroyAllWindows()
    print("Done!")

if __name__ == '__main__':
    task_name = 'isaac_singlecolorbox'
    save_data_path = '/home/cvte/twilight/data/sim_isaac_singlecolorbox'
    collect_data_main(task_name = task_name, save_data_path = save_data_path, total_data_num = 150)

    #collect_data_main(task_name = task_name, total_data_num = 300)