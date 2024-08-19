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

class GripperGearInsertion():
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

        # Default to PhysX
        args.physics_engine = gymapi.SIM_PHYSX
        args.use_gpu = (args.sim_device_type == 'cuda')

        if args.flex:
            args.physics_engine = gymapi.SIM_FLEX

        # Using --nographics implies --headless
        if no_graphics and args.nographics:
            args.headless = True

        if args.slices is None:
            args.slices = args.subscenes

        return args

    def create_gym_env(self,):
        torch.set_printoptions(precision=4, sci_mode=False)

        # acquire gym interface
        self.gym = gymapi.acquire_gym()

        self.controller = "ik"
        self.sim_type = self.args.physics_engine

        # set torch device
        self.device = self.args.sim_device if self.args.use_gpu_pipeline else 'cpu'

        # configure sim
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

        # create sim
        self.sim = self.gym.create_sim(self.args.compute_device_id, self.args.graphics_device_id, self.args.physics_engine, self.sim_params)
        if self.sim is None:
            raise Exception("Failed to create sim")

        # create viewer
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            print("*** Failed to create viewer")
            quit()

        # create table asset
        table_dims = gymapi.Vec3(0.6, 1.0, 0.4)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

        gear_asset_root = "/home/cvte/Documents/IsaacGymEnvs/assets/industreal"
        gear_base_file = "urdf/industreal_gear_base.urdf"
        base_options = gymapi.AssetOptions()
        base_options.flip_visual_attachments = False
        base_options.fix_base_link = True
        base_options.thickness = 0.0  # default = 0.02
        base_options.density = 2700.0  # default = 1000.0
        base_options.armature = 0.0  # default = 0.0
        base_options.use_physx_armature = True
        base_options.linear_damping = 0.0  # default = 0.0
        base_options.max_linear_velocity = 1000.0  # default = 1000.0
        base_options.angular_damping = 0.0  # default = 0.5
        base_options.max_angular_velocity = 64.0  # default = 64.0
        base_options.disable_gravity = False
        base_options.enable_gyroscopic_forces = True
        base_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        base_options.use_mesh_materials = False
        gear_base_asset = self.gym.load_asset(self.sim, gear_asset_root, gear_base_file, base_options)

        gear_options = gymapi.AssetOptions()
        gear_options.flip_visual_attachments = False
        gear_options.fix_base_link = False
        gear_options.thickness = 0.0  # default = 0.02
        gear_options.density = 1000.0  # default = 1000.0
        gear_options.armature = 0.0  # default = 0.0
        gear_options.use_physx_armature = True
        gear_options.linear_damping = 0.5  # default = 0.0
        gear_options.max_linear_velocity = 1000.0  # default = 1000.0
        gear_options.angular_damping = 0.5  # default = 0.5
        gear_options.max_angular_velocity = 64.0  # default = 64.0
        gear_options.disable_gravity = False
        gear_options.enable_gyroscopic_forces = True
        gear_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        gear_options.use_mesh_materials = False
        gear_large_file = "urdf/industreal_gear_large.urdf"
        gear_large_asset = self.gym.load_asset(self.sim, gear_asset_root, gear_large_file, gear_options)
        gear_medium_file = "urdf/industreal_gear_medium.urdf"
        gear_medium_asset = self.gym.load_asset(self.sim, gear_asset_root, gear_medium_file, gear_options)
        gear_small_file = "urdf/industreal_gear_small.urdf"
        gear_small_asset = self.gym.load_asset(self.sim, gear_asset_root, gear_small_file, gear_options)
            
        # load franka asset
        franka_asset_root = '/home/cvte/Documents/isaacgym/assets'
        franka_asset_file = "urdf/franka_description/robots/franka_panda.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = True
        franka_asset = self.gym.load_asset(self.sim, franka_asset_root, franka_asset_file, asset_options)
        self.body_dict = self.gym.get_asset_rigid_body_dict(franka_asset)
        # {'panda_hand': 8, 'panda_leftfinger': 9, 'panda_link0': 0, 'panda_link1': 1, 'panda_link2': 2, 'panda_link3': 3, 'panda_link4': 4, 'panda_link5': 5, 'panda_link6': 6, 'panda_link7': 7, 'panda_rightfinger': 10}

        # configure franka dofs
        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        self.franka_lower_limits = franka_dof_props["lower"]
        self.franka_upper_limits = franka_dof_props["upper"]
        franka_mids = (self.franka_upper_limits + self.franka_lower_limits) / 2
        # Set position drive for all dofs
        franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_POS)
        franka_dof_props["stiffness"][:7].fill(400.0)
        franka_dof_props["damping"][:7].fill(40.0)
        # Joint 7 and 8 are the gripper open and close joints.
        franka_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
        franka_dof_props["stiffness"][7:].fill(800.0)
        franka_dof_props["damping"][7:].fill(40.0)

        # default dof states and position targets
        franka_num_dofs = self.gym.get_asset_dof_count(franka_asset)
        self.default_dof_pos = np.zeros(franka_num_dofs, dtype=np.float32)
        self.default_dof_pos[:7] = franka_mids[:7]
        # grippers open
        self.default_dof_pos[7:] = self.franka_upper_limits[7:]

        default_dof_state = np.zeros(franka_num_dofs, gymapi.DofState.dtype)
        default_dof_state["pos"] = self.default_dof_pos

        # get link index of panda hand, which we will use as end effector
        franka_link_dict = self.gym.get_asset_rigid_body_dict(franka_asset)
        franka_hand_index = franka_link_dict["panda_hand"]

        # configure env grid
        num_envs = self.num_envs
        num_per_row = int(math.sqrt(num_envs))
        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        franka_pose = gymapi.Transform()
        franka_pose.p = gymapi.Vec3(0, 0, 0)

        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.5, 0.0, 0.5 * table_dims.z)

        gear_base_pose = gymapi.Transform()
        gear_large_pose = gymapi.Transform()
        gear_medium_pose = gymapi.Transform()
        gear_small_pose = gymapi.Transform()

        self.envs = []
        self.hand_idxs = []
        self.gear_base_idxs = []
        self.gear_large_idxs = []
        self.gear_medium_idxs = []
        self.gear_small_idxs = []
        self.task_instruction = []

        # add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)

        for i in range(num_envs):
            # create env
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(env)

            # add table
            table_handle = self.gym.create_actor(env, table_asset, table_pose, "table", i, 0)

            # add gear base
            gear_base_pose.p.x = table_pose.p.x + np.random.uniform(-0.15, 0.00)
            gear_base_pose.p.y = table_pose.p.y + np.random.uniform(-0.1, 0.0)
            gear_base_pose.p.z = table_dims.z
            #gear_base_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))
            gear_base_handle = self.gym.create_actor(env, gear_base_asset, gear_base_pose, "gear_base", i, 0)
            gear_base_props = self.gym.get_actor_rigid_shape_properties(env, gear_base_handle)
            gear_base_props[0].friction = 1.0  # default = ?
            gear_base_props[0].rolling_friction = 1.0  # default = 0.0
            gear_base_props[0].torsion_friction = 1.0  # default = 0.0
            gear_base_props[0].restitution = 0.0  # default = ?
            gear_base_props[0].compliance = 0.0  # default = 0.0
            gear_base_props[0].thickness = 0.0  # default = 0.0
            self.gym.set_actor_rigid_shape_properties(env, gear_base_handle, gear_base_props)
            gear_base_idx = self.gym.get_actor_rigid_body_index(env, gear_base_handle, 0, gymapi.DOMAIN_SIM)
            self.gear_base_idxs.append(gear_base_idx)

            # add gear large
            gear_large_pose.p.x = table_pose.p.x + np.random.uniform(-0.15, -0.0)
            gear_large_pose.p.y = table_pose.p.y + np.random.uniform(0.05, 0.4)
            gear_large_pose.p.z = table_dims.z + 0.02
            gear_large_handle = self.gym.create_actor(env, gear_large_asset, gear_large_pose, "gear_large", i, 0)
            gear_large_props = self.gym.get_actor_rigid_shape_properties(env, gear_large_handle)
            gear_large_props[0].friction = 1.0  # default = ?
            gear_large_props[0].rolling_friction = 1.0  # default = 0.0
            gear_large_props[0].torsion_friction = 1.0  # default = 0.0
            gear_large_props[0].restitution = 0.0  # default = ?
            gear_large_props[0].compliance = 0.0  # default = 0.0
            gear_large_props[0].thickness = 0.0  # default = 0.0
            self.gym.set_actor_rigid_shape_properties(env, gear_large_handle, gear_large_props)
            gear_large_idx = self.gym.get_actor_rigid_body_index(env, gear_large_handle, 0, gymapi.DOMAIN_SIM)
            self.gear_large_idxs.append(gear_large_idx)

            # add gear medium
            gear_medium_pose.p.x = table_pose.p.x + np.random.uniform(-0.15, -0.0)
            gear_medium_pose.p.y = table_pose.p.y + np.random.uniform(0.05, 0.4)
            gear_medium_pose.p.z = table_dims.z + 0.02
            gear_medium_handle = self.gym.create_actor(env, gear_medium_asset, gear_medium_pose, "gear_medium", i, 0)
            gear_medium_props = self.gym.get_actor_rigid_shape_properties(env, gear_medium_handle)
            gear_medium_props[0].friction = 1.0  # default = ?
            gear_medium_props[0].rolling_friction = 1.0  # default = 0.0
            gear_medium_props[0].torsion_friction = 1.0  # default = 0.0
            gear_medium_props[0].restitution = 0.0  # default = ?
            gear_medium_props[0].compliance = 0.0  # default = 0.0
            gear_medium_props[0].thickness = 0.0  # default = 0.0
            self.gym.set_actor_rigid_shape_properties(env, gear_medium_handle, gear_medium_props)
            gear_medium_idx = self.gym.get_actor_rigid_body_index(env, gear_medium_handle, 0, gymapi.DOMAIN_SIM)
            self.gear_medium_idxs.append(gear_medium_idx)

            # add gear small
            gear_small_pose.p.x = table_pose.p.x + np.random.uniform(-0.15, -0.0)
            gear_small_pose.p.y = table_pose.p.y + np.random.uniform(0.05, 0.4)
            gear_small_pose.p.z = table_dims.z + 0.02
            gear_small_handle = self.gym.create_actor(env, gear_small_asset, gear_small_pose, "gear_small", i, 0)
            gear_small_props = self.gym.get_actor_rigid_shape_properties(env, gear_small_handle)
            gear_small_props[0].friction = 1.0  # default = ?
            gear_small_props[0].rolling_friction = 1.0  # default = 0.0
            gear_small_props[0].torsion_friction = 1.0  # default = 0.0
            gear_small_props[0].restitution = 0.0  # default = ?
            gear_small_props[0].compliance = 0.0  # default = 0.0
            gear_small_props[0].thickness = 0.0  # default = 0.0
            self.gym.set_actor_rigid_shape_properties(env, gear_small_handle, gear_small_props)
            gear_small_idx = self.gym.get_actor_rigid_body_index(env, gear_small_handle, 0, gymapi.DOMAIN_SIM)
            self.gear_small_idxs.append(gear_small_idx)

            # add franka
            franka_handle = self.gym.create_actor(env, franka_asset, franka_pose, "franka", i, 0)

            # set dof properties
            self.gym.set_actor_dof_properties(env, franka_handle, franka_dof_props)

            # set initial dof states
            self.gym.set_actor_dof_states(env, franka_handle, default_dof_state, gymapi.STATE_ALL)

            # set initial position targets
            self.gym.set_actor_dof_position_targets(env, franka_handle, self.default_dof_pos)

            # get global index of hand in rigid body state tensor
            hand_idx = self.gym.find_actor_rigid_body_index(env, franka_handle, "panda_hand", gymapi.DOMAIN_SIM)
            self.hand_idxs.append(hand_idx)

            # Generate language instruction.
            self.task_instruction.append("Please place the three gears on the gear base.")

        # point camera at middle env
        cam_pos = gymapi.Vec3(4, 3, 2)
        cam_target = gymapi.Vec3(-4, -3, 0)
        middle_env = self.envs[num_envs // 2 + num_per_row // 2]
        self.gym.viewer_camera_look_at(self.viewer, middle_env, cam_pos, cam_target)

        camera_props = copy.deepcopy(gymapi.CameraProperties())
        camera_props.width = 320
        camera_props.height = 240
        self.cameras = []
        for env in self.envs:
            # Top camera
            top_camera = self.gym.create_camera_sensor(env, camera_props)
            camera_pos = gymapi.Vec3(0.5, 0, 1.5)
            camera_target = gymapi.Vec3(0, 0, -1)
            self.gym.set_camera_location(top_camera, env, camera_pos, camera_target)

            # Side camera 1
            side_camera1 = self.gym.create_camera_sensor(env, camera_props)
            camera_pos = gymapi.Vec3(0.5, 0.7, 0.8)
            camera_target = gymapi.Vec3(0, -3, -0.5)
            self.gym.set_camera_location(side_camera1, env, camera_pos, camera_target)

            # Side camera 2
            side_camera2 = self.gym.create_camera_sensor(env, camera_props)
            camera_pos = gymapi.Vec3(0.5, -0.7, 0.8)
            camera_target = gymapi.Vec3(0, 3, -0.5)
            self.gym.set_camera_location(side_camera2, env, camera_pos, camera_target)

            # Hand camera
            hand_camera = self.gym.create_camera_sensor(env, camera_props)
            local_transform = gymapi.Transform()
            local_transform.p = gymapi.Vec3(0.1, 0, 0)
            local_transform.r = gymapi.Quat.from_euler_zyx(0, -90, 0)
            finger_handle = self.gym.find_actor_rigid_body_handle(env, franka_handle, "panda_hand")
            self.gym.attach_camera_to_body(hand_camera, env, finger_handle, local_transform, gymapi.FOLLOW_TRANSFORM)

            self.cameras.append(dict(top_camera = top_camera, side_camera1 = side_camera1, side_camera2 = side_camera2, hand_camera = hand_camera))

        # ==== prepare tensors =====
        # from now on, we will use the tensor API that can run on CPU or GPU
        self.gym.prepare_sim(self.sim)

        # get jacobian tensor
        # for fixed-base franka, tensor has shape (num envs, 10, 6, 9)
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "franka")
        jacobian = gymtorch.wrap_tensor(_jacobian)

        # jacobian entries corresponding to franka hand
        self.j_eef = jacobian[:, franka_hand_index - 1, :, :7]
        
        # get mass matrix tensor
        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "franka")
        mm = gymtorch.wrap_tensor(_massmatrix)
        self.mm = mm[:, :7, :7]          # only need elements corresponding to the franka arm

        # get rigid body state tensor
        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(_rb_states)

        # get dof state tensor
        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        dof_states = gymtorch.wrap_tensor(_dof_states)
        self.dof_pos = dof_states[:, 0].view(num_envs, 9, 1)

        # Set action tensors
        self.pos_action = torch.zeros_like(self.dof_pos).squeeze(-1)
        self.effort_action = torch.zeros_like(self.pos_action)

    def update_simulator_before_ctrl(self):
        # step the physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        # refresh tensors
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

    def update_simulator_after_ctrl(self):
        # step rendering
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
        # solve damped least squares
        kp = 0.2
        j_eef_T = torch.transpose(self.j_eef, 1, 2)
        lmbda = torch.eye(6, device=dpose.device) * (self.damping ** 2)
        u = kp * (j_eef_T @ torch.inverse(self.j_eef @ j_eef_T + lmbda) @ dpose).view(self.num_envs, 7)
        return u
    
    def clean_up(self):
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

if __name__ == '__main__':
    env = GripperGearInsertion(num_envs = 1)
    total_count = 300
    cnt = 0
    while True:
        env.update_simulator_before_ctrl()
        progress = cnt % total_count / total_count
        if progress < 0.3:
            gripper_ctrl = env.franka_upper_limits[7:]
            scroll_ctrl = [env.franka_lower_limits[6],]
        else:
            gripper_ctrl = env.franka_lower_limits[7:]
            scroll_progress = (progress - 0.3) / 7 * 10
            scroll_ctrl = ((env.franka_lower_limits - env.franka_upper_limits) * scroll_progress + env.franka_upper_limits)[6:7]

        default_dof_pos = torch.Tensor(env.default_dof_pos).cuda()
        default_dof_pos[6:7] = torch.Tensor(scroll_ctrl).cuda()
        default_dof_pos[7:9] = torch.Tensor(gripper_ctrl).cuda()
        env.pos_action[0] = default_dof_pos
        env.update_action_map()
        env.update_simulator_after_ctrl()
        cnt += 1
    env.clean_up()