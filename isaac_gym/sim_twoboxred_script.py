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

class TwoBoxRedScript():
    def __init__(self, isaac_env):
        self.isaac_env = isaac_env
        self.box_size = 0.045
        self.down_dir = torch.Tensor([0, 0, -1]).cuda().view(1, 3)
        self.down_q = torch.stack(isaac_env.num_envs * [torch.tensor([1.0, 0.0, 0.0, 0.0])]).cuda().view((isaac_env.num_envs, 4))
        self.grasp_offset = 0.1
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

        print(self.isaac_env.task_instruction)
        self.box_idx_list = []
        for task_instruction, box_idxs in zip(self.isaac_env.task_instruction, self.isaac_env.box_idxs):
            box_key = 'red'
            self.box_idx_list.append(box_idxs[box_key])

    def get_grasp_pos(self,):
        hand_pos = self.isaac_env.rb_states[self.isaac_env.hand_idxs, :3] # Left shape: (num_hand, 3)
        hand_rot = self.isaac_env.rb_states[self.isaac_env.hand_idxs, 3:7]
        hand_vel = self.isaac_env.rb_states[self.isaac_env.hand_idxs, 7:]
        
        box_pos = self.isaac_env.rb_states[self.box_idx_list, :3]  # Left shape: (num_hand, 3)
        box_rot = self.isaac_env.rb_states[self.box_idx_list, 3:7]    # Left shape: (num_hand, 4)
        to_box = box_pos - hand_pos
        box_dist = torch.norm(to_box, dim=-1).unsqueeze(-1)
        box_dir = to_box / box_dist
        box_dot = box_dir @ self.down_dir.view(3, 1)

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
        
        return goal_pos, goal_rot

    def update_pos_action(self,):
        hand_pos = self.isaac_env.rb_states[self.isaac_env.hand_idxs, :3] # Left shape: (num_hand, 3)
        hand_rot = self.isaac_env.rb_states[self.isaac_env.hand_idxs, 3:7]
        hand_vel = self.isaac_env.rb_states[self.isaac_env.hand_idxs, 7:]
        
        box_pos = self.isaac_env.rb_states[self.box_idx_list, :3]  # Left shape: (num_hand, 3)
        box_rot = self.isaac_env.rb_states[self.box_idx_list, 3:7]    # Left shape: (num_hand, 4)
        to_box = box_pos - hand_pos
        box_dist = torch.norm(to_box, dim=-1).unsqueeze(-1)
        box_dir = to_box / box_dist
        box_dot = box_dir @ self.down_dir.view(3, 1)

        if self.action_state == 0:
            # goal action
            goal_pos, goal_rot = self.get_grasp_pos()
            # determine if we're holding the box (grippers are closed and box is near)
            gripper_sep = self.isaac_env.dof_pos[:, 7] + self.isaac_env.dof_pos[:, 8]
            gripped = (gripper_sep < 0.045) & (box_dist < self.grasp_offset + 0.5 * self.box_size)

            # gripper action
            if (box_dist < self.grasp_offset + 0.02) | gripped & torch.logical_not(self.close_gripper):
                self.timer[0] = True
            
            self.close_gripper = (box_dist < self.grasp_offset + 0.02) | gripped
            #grip_acts = torch.where(self.close_gripper, torch.Tensor([[0., 0.]] * self.isaac_env.num_envs).cuda(), torch.Tensor([[0.04, 0.04]] * self.isaac_env.num_envs).cuda())
            grip_acts = torch.Tensor([[0.04, 0.04]] * self.isaac_env.num_envs).cuda()
            self.isaac_env.pos_action[:, 7:9] = grip_acts

            # change action_state
            if self.close_gripper & self.timer[0]:
                self.timer[1] += 1
                if self.timer[1] > 30:
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
            goal_pos = box_pos.new([[self.isaac_env.container_bottom_pose.p.x, self.isaac_env.container_bottom_pose.p.y - 0.02, self.isaac_env.table_dims.z + 0.5 * self.box_size + self.grasp_offset + 0.05],])
            goal_rot = torch.Tensor([0.7070, 0.7070, 0.015, 0.0]).cuda()[None]
            grip_acts = torch.Tensor([[0., 0.]] * self.isaac_env.num_envs).cuda()
            self.isaac_env.pos_action[:, 7:9] = grip_acts
            if box_pos[0, 2] <= (self.isaac_env.table_dims.z + 0.5 * self.box_size + 0.2 + 0.01):
                self.timer[0] = True
                self.timer[1] += 1
                if self.timer[1] > 50:
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

        end_observation = self.isaac_env.rb_states[self.isaac_env.hand_idxs,][0].cpu().numpy().astype(np.float32)    # Left shape: (13,). [0:3]: xyz, [3:7]: rotation quaternion, [7:10]: xyz velocity, [10:13]: rotation velocity.
        joint_observation = self.isaac_env.dof_pos[0, :, 0].cpu().numpy().astype(np.float32)  # Left shape: (9, )
        observation = dict(end_observation = end_observation, joint_observation = joint_observation)
        action = np.concatenate((end_observation[:7], grip_acts[0].cpu().numpy()), axis = 0).astype(np.float32)

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