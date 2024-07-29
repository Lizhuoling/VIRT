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
import matplotlib.pyplot as plt
from itertools import combinations

from gripper_multicolorbox import GripperMultiColorBox
from gripper_singlebox import GripperSingleBox
from gripper_fixedboxes import GripperFixedBoxes
from gripper_singlecolorbox import GripperSingleColorBox
from gripper_twoboxred import GripperTwoBoxRed
from gripper_fiveboxred import GripperFiveBoxRed

def replay(task_name, root_path, start_idx = None):
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
        reward = replay_onecase(task_name, h5py_path)

        if task_name in ['isaac_multicolorbox', 'isaac_fixedboxes']:
            if reward < 3:
                result = 'failure'
                failure_list.append(file_name)
            else:
                result = 'success'
        elif task_name in ['isaac_singlebox', 'isaac_singlecolorbox', 'isaac_twoboxred', 'isaac_fiveboxred']:
            if reward < 1:
                result = 'failure'
                failure_list.append(file_name)
            else:
                result = 'success'
        else:
            raise NotImplementedError
        print(f'{file_name} reward: {reward}, result: {result}')

    print(f"Failure list: {failure_list}")


def replay_onecase(task_name, h5py_path):
    h5f = h5py.File(h5py_path, 'r')
    seed = np.array(h5f['seed']).item()
    actions = h5f['action'][:]  # Left shape: (T, 9)
    h5f.close()

    if task_name == 'isaac_multicolorbox':
        isaac_env = GripperMultiColorBox(num_envs = 1, seed = seed)
    elif task_name == 'isaac_singlebox':
        isaac_env = GripperSingleBox(num_envs = 1, seed = seed)
    elif task_name == 'isaac_fixedboxes':
        isaac_env = GripperFixedBoxes(num_envs = 1, seed = seed)
    elif task_name == 'isaac_singlecolorbox':
        isaac_env = GripperSingleColorBox(num_envs = 1, seed = seed)
    elif task_name == 'isaac_twoboxred':
        isaac_env = GripperTwoBoxRed(num_envs = 1, seed = seed)
    elif task_name == 'isaac_fiveboxred':
        isaac_env = GripperFiveBoxRed(num_envs = 1, seed = seed)

    print(f"Task instruction: {isaac_env.task_instruction[0]}")
    last_time = time.time()
    simulation_step = 0
    last_step = 0
    ctrl_min_step = 2
    action_idx = 0
    while action_idx < actions.shape[0]:
        cur_time = time.time()
        if simulation_step - last_step >= ctrl_min_step:
            step_interval_flag = True
            last_step = simulation_step
        else:
            step_interval_flag = False
        
        isaac_env.update_simulator_before_ctrl()
        if step_interval_flag:
            action = torch.Tensor(actions[action_idx : action_idx + 1]).to(isaac_env.pos_action.device) # Left shape: (1, 9)
            update_pos_action(isaac_env, action)
            isaac_env.update_action_map()
            action_idx += 1
        isaac_env.update_simulator_after_ctrl()
        simulation_step += 1

    if task_name == 'isaac_multicolorbox':
        reward = get_issac_multicolorbox_reward(isaac_env)
    elif task_name == 'isaac_singlebox':
        reward = get_isaac_singlebox_reward(isaac_env)
    elif task_name == 'isaac_fixedboxes':
        reward = get_isaac_fixedboxes_reward(isaac_env)
    elif task_name == 'isaac_singlecolorbox':
        reward = get_isaac_singlecolorbox_reward(isaac_env)
    elif task_name == 'isaac_twoboxred':
        reward = get_isaac_twoboxred_reward(isaac_env)
    elif task_name == 'isaac_fiveboxred':
        reward = get_isaac_fiveboxred_reward(isaac_env)
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

def get_issac_multicolorbox_reward(isaac_env):
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

def get_isaac_singlebox_reward(isaac_env):
    box_idx = isaac_env.box_idxs[0]['red']
    box_xyz = isaac_env.rb_states[box_idx, :3]
    
    if box_xyz[0] > 0.39 and box_xyz[0] < 0.61 and box_xyz[1] > -0.26 and box_xyz[1] < -0.14:
        return 1
    else:
        return 0
    
def get_isaac_singlecolorbox_reward(isaac_env):
    task_instruction = isaac_env.task_instruction[0]
    box_key = task_instruction
    box_idx = isaac_env.box_idxs[0][box_key]
    box_xyz = isaac_env.rb_states[box_idx, :3]
    
    if box_xyz[0] > 0.39 and box_xyz[0] < 0.61 and box_xyz[1] > -0.26 and box_xyz[1] < -0.14:
        return 1
    else:
        return 0
    
def get_isaac_twoboxred_reward(isaac_env):
    task_instruction = 'red'
    box_key = task_instruction
    box_idx = isaac_env.box_idxs[0][box_key]
    box_xyz = isaac_env.rb_states[box_idx, :3]
    
    if box_xyz[0] > 0.39 and box_xyz[0] < 0.61 and box_xyz[1] > -0.26 and box_xyz[1] < -0.14:
        return 1
    else:
        return 0
    
def get_isaac_fiveboxred_reward(isaac_env):
    task_instruction = 'red'
    box_key = task_instruction
    box_idx = isaac_env.box_idxs[0][box_key]
    box_xyz = isaac_env.rb_states[box_idx, :3]
    
    if box_xyz[0] > 0.39 and box_xyz[0] < 0.61 and box_xyz[1] > -0.26 and box_xyz[1] < -0.14:
        return 1
    else:
        return 0
    
def get_isaac_fixedboxes_reward(isaac_env):
    box1_key, box2_key = 'blue', 'red'

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
    replay(task_name = 'isaac_multicolorbox', root_path = '/home/cvte/twilight/data/isaac_multicolorbox/h5py', start_idx = 0)

    #print(replay_onecase(task_name = 'isaac_singlecolorbox', h5py_path = '/home/cvte/twilight/data/isaac_singlecolorbox/h5py/episode_7.hdf5'))