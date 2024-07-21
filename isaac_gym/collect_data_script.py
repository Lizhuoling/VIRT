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
from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt
from itertools import combinations

from gripper_singlebox import GripperSingleBox

class ScriptManager():
    def __init__(self, isaac_env):
        self.isaac_env = isaac_env
        self.box_size = 0.045
        self.down_dir = torch.Tensor([0, 0, -1]).cuda().view(1, 3)
        self.down_q = torch.stack(isaac_env.num_envs * [torch.tensor([1.0, 0.0, 0.0, 0.0])]).cuda().view((isaac_env.num_envs, 4))
        self.grasp_offset = 0.105
        # box corner coords, used to determine grasping yaw
        self.box_half_size = 0.5 * self.box_size
        self.corner_coord = torch.Tensor([self.box_half_size, self.box_half_size, self.box_half_size])
        self.corners = torch.stack(isaac_env.num_envs * [self.corner_coord]).cuda()
        self.timer = [False, 0, False]
        self.close_gripper = torch.tensor(False).cuda()
        self.action_state = 0

        isaac_env.update_simulator_before_ctrl()
        self.init_pos = self.isaac_env.rb_states[self.isaac_env.hand_idxs, :3].clone()
        self.init_rot = self.isaac_env.rb_states[self.isaac_env.hand_idxs, 3:7].clone()
        self.init_pos_ori = self.init_pos.clone()
        self.init_rot_ori = self.init_rot.clone()

    def update_pos_action(self,):
        hand_pos = self.isaac_env.rb_states[self.isaac_env.hand_idxs, :3] # Left shape: (num_hand, 3)
        hand_rot = self.isaac_env.rb_states[self.isaac_env.hand_idxs, 3:7]
        hand_vel = self.isaac_env.rb_states[self.isaac_env.hand_idxs, 7:]
        
        box_pos = self.isaac_env.rb_states[[self.isaac_env.box_idxs[0]['red'],], :3]  # Left shape: (num_hand, 3)
        box_rot = self.isaac_env.rb_states[[self.isaac_env.box_idxs[0]['red'],], 3:7]    # Left shape: (num_hand, 4)
        to_box = box_pos - hand_pos
        box_dist = torch.norm(to_box, dim=-1).unsqueeze(-1)
        box_dir = to_box / box_dist
        box_dot = box_dir @ self.down_dir.view(3, 1)

        end_observation = self.isaac_env.rb_states[self.isaac_env.hand_idxs,][0].cpu().numpy().astype(np.float32)    # Left shape: (13,). [0:3]: xyz, [3:7]: rotation quaternion, [7:10]: xyz velocity, [10:13]: rotation velocity.
        joint_observation = self.isaac_env.dof_pos[0, :, 0].cpu().numpy().astype(np.float32)  # Left shape: (9, )
        observation = dict(end_observation = end_observation, joint_observation = joint_observation)

        if self.action_state == 0:
            # determine if we're holding the box (grippers are closed and box is near)
            gripper_sep = self.isaac_env.dof_pos[:, 7] + self.isaac_env.dof_pos[:, 8]
            gripped = (gripper_sep < 0.045) & (box_dist < self.grasp_offset + 0.5 * self.isaac_env.box_size)
            # if hand is above box, descend to grasp offset
            # otherwise, seek a position above the box
            yaw_q = cube_grasping_yaw(box_rot, self.corners)
            box_yaw_dir = self.isaac_env.quat_axis(yaw_q, 0)
            hand_yaw_dir = self.isaac_env.quat_axis(hand_rot, 0)
            yaw_dot = torch.bmm(box_yaw_dir.view(self.isaac_env.num_envs, 1, 3), hand_yaw_dir.view(self.isaac_env.num_envs, 3, 1)).squeeze(-1)
            above_box = ((box_dot >= 0.99) & (yaw_dot >= 0.95) & (box_dist < self.grasp_offset * 3)).squeeze(-1)
            grasp_pos = box_pos.clone()
            grasp_pos[:, 2] = torch.where(above_box, box_pos[:, 2] + self.grasp_offset, box_pos[:, 2] + self.grasp_offset * 2.5)

            # goal action
            goal_pos = grasp_pos
            goal_rot = quat_mul(self.down_q, quat_conjugate(yaw_q))

            # gripper action
            if (box_dist < self.grasp_offset + 0.02) | gripped & torch.logical_not(self.close_gripper):
                self.timer[0] = True
            
            self.close_gripper = (box_dist < self.grasp_offset + 0.02) | gripped
            grip_acts = torch.where(self.close_gripper, torch.Tensor([[0., 0.]] * self.isaac_env.num_envs).cuda(), torch.Tensor([[0.04, 0.04]] * self.isaac_env.num_envs).cuda())
            self.isaac_env.pos_action[:, 7:9] = grip_acts

            # change action_state
            if self.close_gripper & self.timer[0]:
                self.timer[1] += 1
                if self.timer[1] > 15:
                    self.action_state += 1
                    self.timer[0] = False
                    self.timer[1] = 0
                    self.statue_action2_rot = goal_rot.clone()

        # Grip Object
        elif self.action_state == 1:
            goal_pos = self.init_pos_ori
            goal_rot = self.init_rot_ori
            grip_acts = torch.Tensor([[0., 0.]] * self.isaac_env.num_envs).cuda()
            self.isaac_env.pos_action[:, 7:9] = grip_acts
            if torch.norm(self.init_pos - hand_pos, dim=-1) <= 0.01:
                self.timer[0] = True
                self.timer[1] += 1
                if self.timer[1] > 5:
                    self.action_state += 1
                    self.timer[0] = False
                    self.timer[1] = 0

        # To destination
        elif self.action_state == 2:
            goal_pos = box_pos.new([[self.isaac_env.container_bottom_pose.p.x, self.isaac_env.container_bottom_pose.p.y, self.isaac_env.table_dims.z + 0.5 * self.box_size + self.grasp_offset + 0.2],])
            goal_rot = self.statue_action2_rot
            grip_acts = torch.Tensor([[0., 0.]] * self.isaac_env.num_envs).cuda()
            self.isaac_env.pos_action[:, 7:9] = grip_acts
            if box_pos[0, 2] <= (self.isaac_env.table_dims.z + 0.5 * self.isaac_env.box_size + 0.2 + 0.01):
                self.timer[0] = True
                self.timer[1] += 1
                if self.timer[1] > 20:
                    self.action_state += 1
                    self.timer[0] = False
                    self.timer[1] = 0

        # Release object1
        elif self.action_state == 3:
            goal_pos = self.isaac_env.rb_states[self.isaac_env.hand_idxs, :3].clone()
            goal_rot = self.isaac_env.rb_states[self.isaac_env.hand_idxs, 3:7].clone()
            grip_acts = torch.Tensor([[0.04, 0.04]] * self.isaac_env.num_envs).cuda()
            self.isaac_env.pos_action[:, 7:9] = grip_acts
            self.timer[1] += 1
            if self.timer[1] == 20:
                self.action_state += 1
                self.timer[1] = 0
                self.timer[2] = True

        # IK deploy control
        pos_err = goal_pos - hand_pos
        orn_err = self.isaac_env.orientation_error(goal_rot, hand_rot)
        dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)
        self.isaac_env.pos_action[:, :7] = self.isaac_env.dof_pos.squeeze(-1)[:, :7] + self.isaac_env.control_ik(dpose)
        action = torch.cat((goal_pos, goal_rot, grip_acts), dim = 1)[0].cpu().numpy().astype(np.float32)
        end_observation = self.isaac_env.rb_states[self.isaac_env.hand_idxs,][0].cpu().numpy().astype(np.float32)    # Left shape: (13,). [0:3]: xyz, [3:7]: rotation quaternion, [7:10]: xyz velocity, [10:13]: rotation velocity.
        joint_observation = self.isaac_env.dof_pos[0, :, 0].cpu().numpy().astype(np.float32)  # Left shape: (9, )
        observation = dict(end_observation = end_observation, joint_observation = joint_observation)

        if self.timer[2] == True:
            end_signal = True
        else:
            end_signal = False

        return action, observation, end_signal

def cube_grasping_yaw(q, corners):
    """ returns horizontal rotation required to grasp cube """
    rc = quat_rotate(q, corners)
    yaw = (torch.atan2(rc[:, 1], rc[:, 0]) - 0.25 * math.pi) % (0.5 * math.pi)
    theta = 0.5 * yaw
    w = theta.cos()
    x = torch.zeros_like(w)
    y = torch.zeros_like(w)
    z = theta.sin()
    yaw_quats = torch.stack([x, y, z, w], dim=-1)
    return yaw_quats

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
        script_manager = ScriptManager(isaac_env = isaac_env)

    last_time = time.time()
    ctrl_min_time = 0.05
    print('\033[93m' + "Task instruction: {}".format(isaac_env.task_instruction[0]) + '\033[0m')

    action_list, observations_list, images_list = [], [], []
    end_signal = False
    sample_cnt = 0
    print(f"Start collect sample {sample_cnt}")
    while True:
        cur_time = time.time()
        if cur_time - last_time < ctrl_min_time:
            time_flag = False
        else:
            time_flag = True
            last_time = cur_time
        isaac_env.update_simulator_before_ctrl()

        if time_flag:
            action, observations, end_signal = script_manager.update_pos_action()
            isaac_env.update_action_map()

        isaac_env.update_simulator_after_ctrl()
        images_envs = isaac_env.visualization_process()
        assert len(images_envs) ==  1
        images = images_envs[0]
        
        # Save data.
        if save_data_flag and time_flag:
            action_list.append(action)
            observations_list.append(observations)
            images_list.append(images)

        if end_signal:
            isaac_env.clean_up()
            if save_data_flag:
                save_data_manager.save_data(action_list = action_list, observations_list = observations_list, images_list = images_list, task_instruction = isaac_env.task_instruction[0], seed = isaac_env.random_seed)
            isaac_env.init_simulate_env()
            script_manager = ScriptManager(isaac_env = isaac_env)
            action_list, observations_list, images_list = [], [], []
            sample_cnt += 1
            print(f"Start collect sample {sample_cnt}")

        if sample_cnt >= total_data_num:
            break
            
    isaac_env.clean_up()
    cv2.destroyAllWindows()
    print("Done!")

if __name__ == '__main__':
    task_name = 'isaac_singlebox'
    save_data_path = '/home/cvte/twilight/data/sim_isaac_singlebox'
    collect_data_main(task_name = task_name, save_data_path = save_data_path, total_data_num = 21)