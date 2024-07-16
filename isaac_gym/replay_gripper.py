from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *

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
import matplotlib.pyplot as plt
from itertools import combinations

from gripper_hand import hand_imitate

def replay(root_path, start_idx = None):
    file_name_list = sorted(os.listdir(root_path))
    file_name_list = [ele for ele in file_name_list if ele.endswith('.hdf5')]
    if start_idx != None:
        new_file_name_list = []
        for file_name in file_name_list:
            id = int(file_name[8:-5])
            if id >= start_idx:
                new_file_name_list.append(file_name)
        file_name_list = new_file_name_list

    failure_list = []

    for file_name in file_name_list:
        h5py_path = os.path.join(root_path, file_name)
        reward = replay_onecase(h5py_path)
        if reward < 3:
            result = 'failure'
            failure_list.append(file_name)
        else:
            result = 'success'
        print(f'{file_name} reward: {reward}, result: {result}')
    print(f"Failure list: {failure_list}")


def replay_onecase(h5py_path):
    h5f = h5py.File(h5py_path, 'r')
    seed = np.array(h5f['seed']).item()
    actions = h5f['action'][:]  # Left shape: (T, 9)
    h5f.close()

    isaac_env = hand_imitate(num_envs = 1, seed = seed)

    last_time = time.time()
    ctrl_min_time = 0.05
    action_idx = 0

    while action_idx < actions.shape[0]:
        cur_time = time.time()
        if cur_time - last_time >= ctrl_min_time:
            time_flag = True
            last_time = cur_time
        else:
            time_flag = False
        
        isaac_env.update_simulator_before_ctrl()
        if time_flag:
            action = torch.Tensor(actions[action_idx : action_idx + 1]).to(isaac_env.pos_action.device) # Left shape: (1, 9)
            update_pos_action(isaac_env, action)
            isaac_env.update_action_map()
            #isaac_env.gym.set_dof_position_target_tensor(isaac_env.sim, gymtorch.unwrap_tensor(action))
            action_idx += 1
        isaac_env.update_simulator_after_ctrl()

    reward = get_reward(isaac_env)
    isaac_env.clean_up()

    return reward
    

def update_pos_action(isaac_env, action):
    franka_xyz_range = dict(x_min = 0.2, x_max = 0.8, y_min = -0.5, y_max = 0.5, z_min = 0.4, z_max = 0.8)

    hand_pos = isaac_env.rb_states[isaac_env.hand_idxs, :3]
    hand_rot = isaac_env.rb_states[isaac_env.hand_idxs, 3:7]
    
    goal_pos = action[:, :3]
    goal_rot = action[:, 3:7]
    goal_gripper = action[:, 7:]
    
    pos_err = goal_pos - hand_pos

    orn_err = isaac_env.orientation_error(goal_rot, hand_rot)
    
    dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)
    arm_ctrl = isaac_env.dof_pos.squeeze(-1)[:, :7] + isaac_env.control_ik(dpose)   # Control all joints except the gripper.
    isaac_env.pos_action[:, :7] = arm_ctrl
    isaac_env.pos_action[:, 7:] = goal_gripper

def get_reward(isaac_env):
    assert isaac_env.num_envs == 1

    task_instruction = isaac_env.task_instruction[0]
    task_instruction = task_instruction.split(' ')
    box1_key, box2_key = task_instruction[3], task_instruction[11]

    box1_idx = isaac_env.box_idxs[0][box1_key]
    box2_idx = isaac_env.box_idxs[0][box2_key]
    box1_xyz = isaac_env.rb_states[box1_idx, :3]
    box2_xyz = isaac_env.rb_states[box2_idx, :3]

    reward = 0
    if box1_xyz[0] > 0.39 and box1_xyz[0] < 0.61 and box1_xyz[1] > -0.26 and box1_xyz[1] < -0.14:
        reward += 1
    else:
        return reward
    
    if box2_xyz[0] > 0.39 and box2_xyz[0] < 0.61 and box2_xyz[1] > -0.26 and box2_xyz[1] < -0.14:
        reward += 1

    if torch.norm(box1_xyz[0:2] - box2_xyz[0:2])  < 0.032 and box2_xyz[2] - box1_xyz[2] > 0.022:
        reward += 1
    
    return reward

if __name__ == '__main__':
    #replay(root_path = '/home/cvte/twilight/home/data/own_data/isaac_gripper/h5py', start_idx = 81)

    print(replay_onecase('/home/cvte/twilight/home/data/own_data/isaac_gripper/h5py/episode_94.hdf5'))