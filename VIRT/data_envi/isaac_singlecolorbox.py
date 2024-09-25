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
import argparse

class GripperSingleColorBox():
    def __init__(self, num_envs = 1, seed = None):
        self.num_envs = num_envs

        self.args = self.create_args()
        
        self.create_gym_env()
        self.init_simulate_env(seed = seed)

    def create_args(self, headless=False, no_graphics=False, custom_parameters=[]):
        parser = argparse.ArgumentParser(description="Isaac Gym")
        if headless:
            parser.add_argument('--headless', action='store_true', help='Run headless without creating a viewer window')
        if no_graphics:
            parser.add_argument('--nographics', action='store_true',
                                help='Disable graphics context creation, no viewer window is created, and no headless rendering is available')
        parser.add_argument('--sim_device', type=str, default="cuda:0", help='Physics Device in PyTorch-like syntax')
        parser.add_argument('--pipeline', type=str, default="gpu", help='Tensor API pipeline (cpu/gpu)')
        parser.add_argument('--graphics_device_id', type=int, default=0, help='Graphics Device ID')

        physics_group = parser.add_mutually_exclusive_group()
        physics_group.add_argument('--flex', action='store_true', help='Use FleX for physics')
        physics_group.add_argument('--physx', action='store_true', help='Use PhysX for physics')

        parser.add_argument('--num_threads', type=int, default=0, help='Number of cores used by PhysX')
        parser.add_argument('--subscenes', type=int, default=0, help='Number of PhysX subscenes to simulate in parallel')
        parser.add_argument('--slices', type=int, help='Number of client threads that process env slices')

        args = parser.parse_args([])

        args.sim_device_type, args.compute_device_id = gymutil.parse_device_str(args.sim_device)
        pipeline = args.pipeline.lower()

        assert (pipeline == 'cpu' or pipeline in ('gpu', 'cuda')), f"Invalid pipeline '{args.pipeline}'. Should be either cpu or gpu."
        args.use_gpu_pipeline = (pipeline in ('gpu', 'cuda'))

        if args.sim_device_type != 'cuda' and args.flex:
            print("Can't use Flex with CPU. Changing sim device to 'cuda:0'")
            args.sim_device = 'cuda:0'
            args.sim_device_type, args.compute_device_id = gymutil.parse_device_str(args.sim_device)

        if (args.sim_device_type != 'cuda' and pipeline == 'gpu'):
            print("Can't use GPU pipeline with CPU Physics. Changing pipeline to 'CPU'.")
            args.pipeline = 'CPU'
            args.use_gpu_pipeline = False

        
        args.physics_engine = gymapi.SIM_PHYSX
        args.use_gpu = (args.sim_device_type == 'cuda')

        if args.flex:
            args.physics_engine = gymapi.SIM_FLEX

        
        if no_graphics and args.nographics:
            args.headless = True

        if args.slices is None:
            args.slices = args.subscenes

        return args

    def create_gym_env(self,):
        torch.set_printoptions(precision=4, sci_mode=False)

        
        self.gym = gymapi.acquire_gym()

        self.controller = "ik"
        self.sim_type = self.args.physics_engine

        
        self.device = self.args.sim_device if self.args.use_gpu_pipeline else 'cpu'

        
        self.sim_params = gymapi.SimParams()
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        self.sim_params.dt = 1.0 / 60.0
        self.sim_params.substeps = 2
        self.sim_params.use_gpu_pipeline = self.args.use_gpu_pipeline
        if self.args.physics_engine == gymapi.SIM_PHYSX:
            self.sim_params.physx.solver_type = 1
            self.sim_params.physx.num_position_iterations = 8
            self.sim_params.physx.num_velocity_iterations = 1
            self.sim_params.physx.rest_offset = 0.0
            self.sim_params.physx.contact_offset = 0.001
            self.sim_params.physx.friction_offset_threshold = 0.001
            self.sim_params.physx.friction_correlation_distance = 0.0005
            self.sim_params.physx.num_threads = self.args.num_threads
            self.sim_params.physx.use_gpu = self.args.use_gpu
        else:
            raise Exception("This example can only be used with PhysX")
        
    def set_seed(self, seed):
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

    def init_simulate_env(self, seed = None):
        if seed == None:
            self.random_seed = random.randint(0, 1e5)
        else:
            self.random_seed = seed
        self.set_seed(self.random_seed)

        self.damping = 0.05
        self.recoding_data_flag = False

        
        self.sim = self.gym.create_sim(self.args.compute_device_id, self.args.graphics_device_id, self.args.physics_engine, self.sim_params)
        if self.sim is None:
            raise Exception("Failed to create sim")

        
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            print("*** Failed to create viewer")
            quit()

        
        table_dims = gymapi.Vec3(0.6, 1.0, 0.4)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

        
        container_bottom_dims = gymapi.Vec3(0.3, 0.2, 0.01)
        container_front_dims  = gymapi.Vec3(0.02, 0.2, 0.05)
        container_back_dims   = gymapi.Vec3(0.02, 0.2, 0.05)
        container_left_dims   = gymapi.Vec3(0.26, 0.02, 0.05)
        container_right_dims  = gymapi.Vec3(0.26, 0.02, 0.05)
        container_front_pose_offset  = gymapi.Vec3(-0.14, 0, 0.03)
        container_back_pose_offset   = gymapi.Vec3(0.14, 0, 0.03)
        container_left_pose_offset   = gymapi.Vec3(0, -0.09, 0.03)
        container_right_pose_offset  = gymapi.Vec3(0, 0.09, 0.03)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        container_bottom_asset = self.gym.create_box(self.sim, container_bottom_dims.x, container_bottom_dims.y, container_bottom_dims.z, asset_options)
        container_front_asset  = self.gym.create_box(self.sim, container_front_dims.x, container_front_dims.y, container_front_dims.z, asset_options)
        container_back_asset   = self.gym.create_box(self.sim, container_back_dims.x, container_back_dims.y, container_back_dims.z, asset_options)
        container_left_asset   = self.gym.create_box(self.sim, container_left_dims.x, container_left_dims.y, container_left_dims.z, asset_options)
        container_right_asset  = self.gym.create_box(self.sim, container_right_dims.x, container_right_dims.y, container_right_dims.z, asset_options)

        
        box_size = 0.045
        asset_options = gymapi.AssetOptions()
        box_asset = self.gym.create_box(self.sim, box_size, box_size, box_size, asset_options)
            
        
        asset_root = "/home/cvte/Documents/isaacgym/assets"
        franka_asset_file = "urdf/franka_description/robots/franka_panda.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = True
        franka_asset = self.gym.load_asset(self.sim, asset_root, franka_asset_file, asset_options)
        self.body_dict = self.gym.get_asset_rigid_body_dict(franka_asset)
        

        
        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        self.franka_lower_limits = franka_dof_props["lower"]
        self.franka_upper_limits = franka_dof_props["upper"]
        franka_mids = 0.3 * (self.franka_upper_limits + self.franka_lower_limits)
        
        franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_POS)
        franka_dof_props["stiffness"][:7].fill(400.0)
        franka_dof_props["damping"][:7].fill(40.0)
        
        franka_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
        franka_dof_props["stiffness"][7:].fill(800.0)
        franka_dof_props["damping"][7:].fill(40.0)

        
        franka_num_dofs = self.gym.get_asset_dof_count(franka_asset)
        default_dof_pos = np.zeros(franka_num_dofs, dtype=np.float32)
        default_dof_pos[:7] = franka_mids[:7]
        
        default_dof_pos[7:] = self.franka_upper_limits[7:]

        default_dof_state = np.zeros(franka_num_dofs, gymapi.DofState.dtype)
        default_dof_state["pos"] = default_dof_pos

        
        franka_link_dict = self.gym.get_asset_rigid_body_dict(franka_asset)
        franka_hand_index = franka_link_dict["panda_hand"]

        
        num_envs = self.num_envs
        num_per_row = int(math.sqrt(num_envs))
        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        franka_pose = gymapi.Transform()
        franka_pose.p = gymapi.Vec3(0, 0, 0)

        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.5, 0.0, 0.5 * table_dims.z)

        container_bottom_pose = gymapi.Transform()
        container_front_pose  = gymapi.Transform()
        container_back_pose   = gymapi.Transform()
        container_left_pose   = gymapi.Transform()
        container_right_pose  = gymapi.Transform()
        container_bottom_pose.p = gymapi.Vec3(table_pose.p.x, table_pose.p.y - 0.2, table_dims.z + 0.5 * container_bottom_dims.z)
        container_front_pose.p  = gymapi.Vec3(container_bottom_pose.p.x + container_front_pose_offset.x, 
                                            container_bottom_pose.p.y + container_front_pose_offset.y,
                                            container_bottom_pose.p.z + container_front_pose_offset.z)
        container_back_pose.p   = gymapi.Vec3(container_bottom_pose.p.x + container_back_pose_offset.x, 
                                            container_bottom_pose.p.y + container_back_pose_offset.y,
                                            container_bottom_pose.p.z + container_back_pose_offset.z)
        container_left_pose.p   = gymapi.Vec3(container_bottom_pose.p.x + container_left_pose_offset.x, 
                                            container_bottom_pose.p.y + container_left_pose_offset.y,
                                            container_bottom_pose.p.z + container_left_pose_offset.z)
        container_right_pose.p  = gymapi.Vec3(container_bottom_pose.p.x + container_right_pose_offset.x, 
                                            container_bottom_pose.p.y + container_right_pose_offset.y,
                                            container_bottom_pose.p.z + container_right_pose_offset.z)
        
        self.container_bottom_pose = container_bottom_pose
        self.table_dims = table_dims

        boxes_pose = []
        box_num = 5
        for i in range(box_num):
            boxes_pose.append(gymapi.Transform())

        self.envs = []
        self.box_idxs = []
        self.hand_idxs = []
        self.task_instruction = []

        
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)

        for i in range(num_envs):
            
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(env)

            
            table_handle = self.gym.create_actor(env, table_asset, table_pose, "table", i, 0)

            
            container_bottom_handle = self.gym.create_actor(env, container_bottom_asset, container_bottom_pose, "container_bottom", i, 0)
            container_front_handle = self.gym.create_actor(env, container_front_asset, container_front_pose, "container_front", i, 0)
            container_back_handle = self.gym.create_actor(env, container_back_asset, container_back_pose, "container_back", i, 0)
            container_left_handle = self.gym.create_actor(env, container_left_asset, container_left_pose, "container_left", i, 0)
            container_right_handle = self.gym.create_actor(env, container_right_asset, container_right_pose, "container_right", i, 0)
            for container_handle in [container_bottom_handle, container_front_handle, container_back_handle, container_left_handle, container_right_handle]:
                self.gym.set_rigid_body_color(env, container_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(np.array(0.2), np.array(0.2), np.array(0.2)))

            
            rerange_flag = True
            while rerange_flag:
                stack_groups = []
                for cnt, box_pose in enumerate(boxes_pose):
                    box_pose.p.x = table_pose.p.x + np.random.uniform(-0.15, 0.05)
                    box_pose.p.y = table_pose.p.y + np.random.uniform(0.05, 0.3)
                    box_pose.p.z = table_dims.z + 0.5 * box_size
                    stack_groups.append([[box_pose, cnt]])
                rerange_flag = False
                for p1, p2, in combinations(boxes_pose, 2):
                    if (np.sum(np.square(np.array([p1.p.x - p2.p.x, p1.p.y - p2.p.y]))) < 2 * (box_size + 0.013) ** 2):
                        rerange_flag = True
                        break
            box_handles = [self.gym.create_actor(env, box_asset, box_pose, "box{}".format(box_cnt), i, 0) for box_cnt, box_pose in enumerate(boxes_pose)]
            box_colors = {
                'red': gymapi.Vec3(1, 0, 0),
                'green': gymapi.Vec3(0, 1, 0),
                'blue': gymapi.Vec3(0, 0, 1),
                'yellow': gymapi.Vec3(1, 1, 0),
                'purple': gymapi.Vec3(1, 0, 1),
            }
            assert len(box_handles) == len(box_colors)
            env_box_idxs_dict = {}
            for box_color, box_handle in zip(box_colors, box_handles):
                self.gym.set_rigid_body_color(env, box_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, box_colors[box_color])
                box_idx = self.gym.get_actor_rigid_body_index(env, box_handle, 0, gymapi.DOMAIN_SIM)
                env_box_idxs_dict[box_color] = box_idx
            self.box_idxs.append(env_box_idxs_dict)

            
            franka_handle = self.gym.create_actor(env, franka_asset, franka_pose, "franka", i, 2)

            
            self.gym.set_actor_dof_properties(env, franka_handle, franka_dof_props)

            
            self.gym.set_actor_dof_states(env, franka_handle, default_dof_state, gymapi.STATE_ALL)

            
            self.gym.set_actor_dof_position_targets(env, franka_handle, default_dof_pos)

            
            hand_idx = self.gym.find_actor_rigid_body_index(env, franka_handle, "panda_hand", gymapi.DOMAIN_SIM)
            self.hand_idxs.append(hand_idx)

            
            self.box_sample_ids = random.sample(range(0, box_num), 1)
            box_color_list = list(box_colors.keys())
            self.task_instruction.append("{}".format(box_color_list[self.box_sample_ids[0]],))

        
        cam_pos = gymapi.Vec3(4, 3, 2)
        cam_target = gymapi.Vec3(-4, -3, 0)
        middle_env = self.envs[num_envs // 2 + num_per_row // 2]
        self.gym.viewer_camera_look_at(self.viewer, middle_env, cam_pos, cam_target)

        camera_props = copy.deepcopy(gymapi.CameraProperties())
        camera_props.width = 320
        camera_props.height = 240
        self.cameras = []
        for env in self.envs:
            
            top_camera = self.gym.create_camera_sensor(env, camera_props)
            camera_pos = gymapi.Vec3(0.5, 0, 1.5)
            camera_target = gymapi.Vec3(0, 0, -1)
            self.gym.set_camera_location(top_camera, env, camera_pos, camera_target)

            
            side_camera1 = self.gym.create_camera_sensor(env, camera_props)
            camera_pos = gymapi.Vec3(0.5, 0.7, 0.8)
            camera_target = gymapi.Vec3(0, -3, -0.5)
            self.gym.set_camera_location(side_camera1, env, camera_pos, camera_target)

            
            side_camera2 = self.gym.create_camera_sensor(env, camera_props)
            camera_pos = gymapi.Vec3(0.5, -0.7, 0.8)
            camera_target = gymapi.Vec3(0, 3, -0.5)
            self.gym.set_camera_location(side_camera2, env, camera_pos, camera_target)

            
            hand_camera = self.gym.create_camera_sensor(env, camera_props)
            local_transform = gymapi.Transform()
            local_transform.p = gymapi.Vec3(0.1, 0, 0)
            local_transform.r = gymapi.Quat.from_euler_zyx(0, -90, 0)
            finger_handle = self.gym.find_actor_rigid_body_handle(env, franka_handle, "panda_hand")
            self.gym.attach_camera_to_body(hand_camera, env, finger_handle, local_transform, gymapi.FOLLOW_TRANSFORM)

            self.cameras.append(dict(top_camera = top_camera, side_camera1 = side_camera1, side_camera2 = side_camera2, hand_camera = hand_camera))

        
        
        self.gym.prepare_sim(self.sim)

        
        
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "franka")
        jacobian = gymtorch.wrap_tensor(_jacobian)

        
        self.j_eef = jacobian[:, franka_hand_index - 1, :, :7]
        
        
        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "franka")
        mm = gymtorch.wrap_tensor(_massmatrix)
        self.mm = mm[:, :7, :7]          

        
        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(_rb_states)

        
        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        dof_states = gymtorch.wrap_tensor(_dof_states)
        self.dof_pos = dof_states[:, 0].view(num_envs, 9, 1)

        
        self.pos_action = torch.zeros_like(self.dof_pos).squeeze(-1)
        self.effort_action = torch.zeros_like(self.pos_action)

    def update_simulator_before_ctrl(self):
        
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

    def update_simulator_after_ctrl(self):
        
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, False)
        self.gym.sync_frame_time(self.sim)
    
    def update_action_map(self):
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.pos_action))
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.effort_action))

    def visualization_process(self):
        images_envs = []
        for cnt, env in enumerate(self.envs):
            top_rgb_image = self.gym.get_camera_image(self.sim, env, self.cameras[cnt]['top_camera'], gymapi.IMAGE_COLOR)
            top_rgb_image = top_rgb_image.reshape(240, 320, 4)[:, :, :3]
            top_bgr_image = cv2.cvtColor(top_rgb_image, cv2.COLOR_RGBA2BGR)

            side1_rgb_image = self.gym.get_camera_image(self.sim, env, self.cameras[cnt]['side_camera1'], gymapi.IMAGE_COLOR)
            side1_rgb_image = side1_rgb_image.reshape(240, 320, 4)[:, :, :3]
            side1_bgr_image = cv2.cvtColor(side1_rgb_image, cv2.COLOR_RGBA2BGR)

            side2_rgb_image = self.gym.get_camera_image(self.sim, env, self.cameras[cnt]['side_camera2'], gymapi.IMAGE_COLOR)
            side2_rgb_image = side2_rgb_image.reshape(240, 320, 4)[:, :, :3]
            side2_bgr_image = cv2.cvtColor(side2_rgb_image, cv2.COLOR_RGBA2BGR)

            hand_rgb_image = self.gym.get_camera_image(self.sim, env, self.cameras[cnt]['hand_camera'], gymapi.IMAGE_COLOR)
            hand_rgb_image = hand_rgb_image.reshape(240, 320, 4)[:, :, :3]
            hand_bgr_image = cv2.cvtColor(hand_rgb_image, cv2.COLOR_RGBA2BGR)

            vis_frame = np.concatenate((top_bgr_image, side1_bgr_image, side2_bgr_image, hand_bgr_image), axis = 1)
            cv2.imshow('vis', vis_frame)

            images = dict(
                top_bgr_image = top_bgr_image,
                side1_bgr_image = side1_bgr_image,
                side2_bgr_image = side2_bgr_image,
                hand_bgr_image = hand_bgr_image
            )
            images_envs.append(images)

        self.gym.end_access_image_tensors(self.sim)
        return images_envs
    
    def quat_axis(self, q, axis=0):
        basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
        basis_vec[:, axis] = 1
        return quat_rotate(q, basis_vec)

    def orientation_error(self, desired, current):
        cc = quat_conjugate(current)
        q_r = quat_mul(desired, cc)
        return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

    def control_ik(self, dpose):
        
        kp = 0.2
        j_eef_T = torch.transpose(self.j_eef, 1, 2)
        lmbda = torch.eye(6, device=dpose.device) * (self.damping ** 2)
        u = kp * (j_eef_T @ torch.inverse(self.j_eef @ j_eef_T + lmbda) @ dpose).view(self.num_envs, 7)
        return u
    
    def clean_up(self):
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

if __name__ == '__main__':
    envi = GripperSingleColorBox()