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

import leap_motion
from gripper_multicolorbox import GripperMultiColorBox
from gripper_singlebox import GripperSingleBox
from gripper_fixedboxes import GripperFixedBoxes
from gripper_singlecolorbox import GripperSingleColorBox
from gripper_fiveboxred import GripperFiveBoxRed

leap_mode = 'right'
leap_conf_thre = 0.3
palm_xyz_range = dict(x_min = -150, x_max = 150, y_min = 150, y_max = 600, z_min = -150, z_max = 150)
franka_xyz_range = dict(x_min = 0.2, x_max = 0.8, y_min = -0.5, y_max = 0.5, z_min = 0.4, z_max = 0.8)
    
class TrajectoryFilter9D:
    def __init__(self):
        self.kf = KalmanFilter(dim_x=18, dim_z=9)

        self.dt = 0.1
        self.kf.F = np.eye(18)
        for i in range(9):
            self.kf.F[i, i+9] = self.dt

        self.kf.H = np.zeros((9, 18))
        self.kf.H[:9, :9] = np.eye(9)
        R_factor = 0.5 # Increase this number can make the filter result more smooth.
        self.kf.R = np.eye(9) * R_factor
        q_process_noise = 0.1   # Decrease this number can make the filter result more smooth.
        self.kf.Q = np.eye(18) * q_process_noise
        self.kf.x = np.zeros(18)
        self.kf.P = np.eye(18)

        self.initial_flag = False

    def set_initial_state(self, x):
        assert x.shape[0] == 9
        self.kf.x[:9] = x

    def filter(self, action):
        self.kf.predict()
        self.kf.update(action)
        return self.kf.x[:9]


def update_pos_action(isaac_env, kf = None):
    global leap_mode, leap_conf_thre, palm_xyz_range, franka_xyz_range 

    hand_pos = isaac_env.rb_states[isaac_env.hand_idxs, :3]
    hand_rot = isaac_env.rb_states[isaac_env.hand_idxs, 3:7]
    hand = leap_motion.get_hand(mode = leap_mode, conf_thre = leap_conf_thre)

    if hand == None: return None
    hand_palm_direction = [hand.palm.direction.x, -hand.palm.direction.y, hand.palm.direction.z]
    hand_palm_normal = [hand.palm.normal.x, -hand.palm.normal.y, hand.palm.normal.z]
    hand_orientation = leap_motion.vector_to_quaternion(hand_palm_direction, hand_palm_normal)

    hand_x, hand_y, hand_z = hand.palm.position.x, hand.palm.position.y, hand.palm.position.z
    hand_x = np.clip(hand_x, palm_xyz_range['x_min'], palm_xyz_range['x_max'])
    hand_y = np.clip(hand_y, palm_xyz_range['y_min'], palm_xyz_range['y_max'])
    hand_z = np.clip(hand_z, palm_xyz_range['z_min'], palm_xyz_range['z_max'])
    hand_x = 1 - (hand_x - palm_xyz_range['x_min']) / (palm_xyz_range['x_max'] - palm_xyz_range['x_min'])   # Correspond to the left-right axis of robot hand.
    hand_y = (hand_y - palm_xyz_range['y_min']) / (palm_xyz_range['y_max'] - palm_xyz_range['y_min'])   # Correspond to the up-down axis of robot hand.
    hand_z = (hand_z - palm_xyz_range['z_min']) / (palm_xyz_range['z_max'] - palm_xyz_range['z_min'])    # Correspond to the forward-backward axis of robot hand.
    
    ctrl_x = hand_x * (franka_xyz_range['x_max'] - franka_xyz_range['x_min']) + franka_xyz_range['x_min']
    ctrl_y = hand_z * (franka_xyz_range['y_max'] - franka_xyz_range['y_min']) + franka_xyz_range['y_min']
    ctrl_z = hand_y * (franka_xyz_range['z_max'] - franka_xyz_range['z_min']) + franka_xyz_range['z_min']

    pinch_strength = 1 - hand.pinch_strength
    pinch_strength = np.clip((pinch_strength - 0.5) / (1 - 0.5), 0, 1)
    gripper_ctrl = pinch_strength * (isaac_env.franka_upper_limits[7:] - isaac_env.franka_lower_limits[7:]) + isaac_env.franka_lower_limits[7:]

    action = np.concatenate((np.array([ctrl_x, ctrl_y, ctrl_z]), hand_orientation.elements, gripper_ctrl), axis = 0)
    if kf != None:
        if kf.initial_flag == False:
            kf.set_initial_state(action)
            filter_action = action
            kf.initial_flag = True
        else:
            filter_action = kf.filter(action)
    else:
        filter_action = action

    goal_pos = torch.Tensor(filter_action[:3][None]).to(hand_pos.device)
    goal_rot = torch.Tensor(filter_action[3:7][None]).to(hand_rot.device)
    goal_gripper = torch.Tensor(filter_action[7:][None]).to(hand_rot.device) 
    
    pos_err = goal_pos - hand_pos
    orn_err = isaac_env.orientation_error(goal_rot, hand_rot)
    
    dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)
    arm_ctrl = isaac_env.dof_pos.squeeze(-1)[:, :7] + isaac_env.control_ik(dpose)   # Control all joints except the gripper.
    isaac_env.pos_action[:, :7] = arm_ctrl
    isaac_env.pos_action[:, 7:] = goal_gripper

    #print_hand_rot = np.array(Quaternion(hand_rot[0].cpu().numpy()).yaw_pitch_roll[::-1]) / math.pi * 180
    #print_goal_rot = np.array(Quaternion(goal_rot[0].cpu().numpy()).yaw_pitch_roll[::-1]) / math.pi * 180
    #print('hand_rot: {}, goal_rot: {}'.format(print_hand_rot, print_goal_rot))
    #print("Goal: {}, Current: {}".format(goal_pos[0], hand_pos[0]))
    end_observation = isaac_env.rb_states[isaac_env.hand_idxs,][0].cpu().numpy().astype(np.float32)    # Left shape: (13,). [0:3]: xyz, [3:7]: rotation quaternion, [7:10]: xyz velocity, [10:13]: rotation velocity.
    joint_observation = isaac_env.dof_pos[0, :, 0].cpu().numpy().astype(np.float32)  # Left shape: (9, )
    observation = dict(end_observation = end_observation, joint_observation = joint_observation)
    
    return filter_action.astype(np.float32), observation

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

def collect_data_main(task_name, save_data_path = ""):
    global leap_mode, leap_conf_thre

    save_data_flag = False
    if save_data_path != "":
        save_data_flag = True

    if save_data_flag:
        save_data_manager = SaveDataManager(save_data_path)

    # Initialize leap motion controller.
    my_listener = leap_motion.LeapListener()
    leap_connection = leap_motion.leap.Connection()
    leap_connection.add_listener(my_listener)
    
    if task_name == 'isaac_multicolorbox':
        isaac_env = GripperMultiColorBox(num_envs = 1)
    elif task_name == 'isaac_singlebox':
        isaac_env = GripperSingleBox(num_envs = 1)
    elif task_name == 'isaac_fixedboxes':
        isaac_env = GripperFixedBoxes(num_envs = 1)
    elif task_name == 'isaac_singlecolorbox':
        isaac_env = GripperSingleColorBox(num_envs = 1)
    elif task_name == 'isaac_fiveboxred':
        isaac_env = GripperFiveBoxRed(num_envs = 1)
    kf = TrajectoryFilter9D()

    simulation_step = 0
    last_step = 0
    ctrl_min_step = 2
    print('\033[93m' + "Task instruction: {}".format(isaac_env.task_instruction[0]) + '\033[0m')
    with leap_connection.open():
            leap_connection.set_tracking_mode(leap_motion.leap.TrackingMode.Desktop)
            while True:
                if simulation_step - last_step < ctrl_min_step:
                    step_flag = False
                else:
                    step_flag = True
                    last_step = simulation_step
                isaac_env.update_simulator_before_ctrl()
                leap_flag = leap_motion.get_hand(mode = leap_mode, conf_thre = leap_conf_thre) != None
                if leap_flag:
                    ego = update_pos_action(isaac_env, kf)
                    if step_flag and ego != None:
                        isaac_env.update_action_map()
                        action, observations = ego

                isaac_env.update_simulator_after_ctrl()
                images_envs = isaac_env.visualization_process()
                assert len(images_envs) ==  1
                images = images_envs[0]
                
                # Save data.
                if save_data_flag and leap_flag and step_flag and ego != None and isaac_env.recoding_data_flag:
                    action_list.append(action)
                    observations_list.append(observations)
                    images_list.append(images)
                    
                key = cv2.waitKey(1)
                if key == ord('q'):
                    print('Quit')
                    break
                elif key == ord('r'):
                    isaac_env.clean_up()
                    isaac_env.init_simulate_env()
                    print('\033[93m' + "Task instruction: {}".format(isaac_env.task_instruction[0]) + '\033[0m')
                elif key == ord('b'):
                    isaac_env.recoding_data_flag = True
                    action_list, observations_list, images_list = [], [], []
                    print('Begin recoding data episode {}...'.format(save_data_manager.episode_index))
                elif key == ord('s'):
                    isaac_env.clean_up()
                    save_data_manager.save_data(action_list = action_list, observations_list = observations_list, images_list = images_list, task_instruction = isaac_env.task_instruction[0], seed = isaac_env.random_seed)
                    isaac_env.init_simulate_env()
                    print('\033[93m' + "Task instruction: {}".format(isaac_env.task_instruction[0]) + '\033[0m')

                simulation_step += 1

    isaac_env.clean_up()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    task_name = 'isaac_multicolorbox'
    save_data_path = '/home/cvte/twilight/data/isaac_multicolorbox'
    collect_data_main(task_name = task_name, save_data_path = save_data_path)

    #collect_data_main(task_name = task_name)

    '''save_data_manager = SaveDataManager(save_data_path)
    action_list = [np.zeros((7, ), dtype = np.float32)]
    observations_list = [dict(end_observation = np.zeros((13, ), dtype = np.float32), joint_observation = np.zeros((9, ), dtype = np.float32))]
    images_list = [
        {
            'top_bgr_image': np.zeros((240, 320, 3), dtype= np.uint8),
            'side1_bgr_image': np.zeros((240, 320, 3), dtype= np.uint8),
            'side2_bgr_image': np.zeros((240, 320, 3), dtype= np.uint8),
            'hand_bgr_image': np.zeros((240, 320, 3), dtype= np.uint8),
        }
    ]
    task_instruction = 'debug'
    seed = -1
    save_data_manager.save_data(action_list, observations_list, images_list, task_instruction, seed)'''