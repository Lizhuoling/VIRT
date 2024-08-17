from isaacgym import gymapi
import pdb
import math
import numpy as np
import time
import cv2
import torch
from torchvision.transforms import functional as F

from isaac_gym.gripper_multicolorbox import GripperMultiColorBox

class IsaacMultiColorBoxTestEnviManager():
    def __init__(self, cfg, policy, stats):
        self.cfg = cfg
        self.policy = policy
        self.stats = stats

        basic_num_envi_per_batch = self.cfg['EVAL']['TEST_ENVI_NUM'] // self.cfg['EVAL']['TEST_ENVI_BATCH_NUM']
        self.num_envi_per_batch_list = [basic_num_envi_per_batch for i in range(self.cfg['EVAL']['TEST_ENVI_BATCH_NUM'])]
        self.num_envi_per_batch_list[-1] += self.cfg['EVAL']['TEST_ENVI_NUM'] % self.cfg['EVAL']['TEST_ENVI_BATCH_NUM']

    def inference(self,):
        rewards = np.zeros((self.cfg['EVAL']['TEST_ENVI_NUM'],), dtype = np.float32)
        with torch.no_grad():
            envi_start_idx = 0
            for batch_idx in range(self.cfg['EVAL']['TEST_ENVI_BATCH_NUM']):
                print("Start inference batch {}...".format(batch_idx))
                isaac_envi = GripperMultiColorBox(num_envs = self.num_envi_per_batch_list[batch_idx])
                print(isaac_envi.task_instruction)

                # Init the envi for one second
                init_start_time = time.time()
                while time.time() - init_start_time <= 1:
                    isaac_envi.update_simulator_before_ctrl()
                    isaac_envi.update_simulator_after_ctrl()

                cur_status = torch.zeros((len(isaac_envi.hand_idxs),), dtype = torch.long).cuda()
                enb_obs_list = []
                joint_obs_list = []
                action_list = []

                actions_pred = None
                simulation_step = 0
                last_simulation_step = 0
                execution_step = 0
                while simulation_step <= self.cfg['EVAL']['INFERENCE_MAX_STEPS']:
                    isaac_envi.update_simulator_before_ctrl()

                    if actions_pred == None or execution_step >= actions_pred.shape[1]:
                        if len(action_list) == 0:
                            norm_end_observation, norm_joint_observation, all_cam_images = self.get_observation(isaac_envi)
                            enb_obs_list.append(norm_end_observation)
                            joint_obs_list.append(norm_joint_observation)
                            action_list.append(torch.cat((norm_end_observation[:, 0:7], norm_joint_observation[:, 7:9]), dim = 1)) # Initialize the first element with joint observation
                        all_cam_images, past_action, end_obs, joint_obs, past_action_is_pad, observation_is_pad, task_instruction, status_pred = self.prepare_policy_input(enb_obs_list, \
                                                                                joint_obs_list, action_list, all_cam_images, isaac_envi.task_instruction, cur_status)
                        norm_actions_pred, status_pred = self.policy(image = all_cam_images, past_action = past_action, end_obs = end_obs, joint_obs = joint_obs, observation_is_pad = observation_is_pad, \
                                    past_action_is_pad = past_action_is_pad, task_instruction_list = task_instruction, status = status_pred)  # Left shape: (num_env, T, 9)
                        action_mean, action_std = self.stats['action_mean'][None, None].to(all_cam_images.device), self.stats['action_std'][None, None].to(all_cam_images.device)
                        actions_pred = norm_actions_pred * action_std + action_mean
                        execution_step = 0
                        cur_status = status_pred.clone()
                    
                    if simulation_step - last_simulation_step >= self.cfg['EVAL']['CTRL_STEP_INTERVAL']:
                        # Save observation data
                        norm_end_observation, norm_joint_observation, all_cam_images = self.get_observation(isaac_envi)
                        enb_obs_list.append(norm_end_observation)
                        joint_obs_list.append(norm_joint_observation)
                        action_list.append(norm_actions_pred[:, execution_step])
                        # Execute an action
                        action = actions_pred[:, execution_step]
                        self.execute_action(action, isaac_envi)
                        isaac_envi.update_action_map()
                        execution_step += 1
                        last_simulation_step = simulation_step

                    isaac_envi.update_simulator_after_ctrl()
                    simulation_step += 1

                reward = self.get_reward(isaac_envi)
                rewards[envi_start_idx :  envi_start_idx + self.num_envi_per_batch_list[batch_idx]] = reward.cpu().numpy()
                envi_start_idx += self.num_envi_per_batch_list[batch_idx]
                isaac_envi.clean_up()
        
        reward0_ratio = (rewards == 0).sum() / rewards.shape[0]
        reward1_ratio = (rewards == 1).sum() / rewards.shape[0]
        reward2_ratio = (rewards == 2).sum() / rewards.shape[0]
        reward3_ratio = (rewards == 3).sum() / rewards.shape[0]
        success_rate = reward3_ratio
        average_reward = np.mean(rewards)
        reward_info = dict(
            reward0_ratio = reward0_ratio,
            reward1_ratio = reward1_ratio,
            reward2_ratio = reward2_ratio,
            reward3_ratio = reward3_ratio,
            success_rate = success_rate,
            average_reward = average_reward
        )
        print(f'\nreward0_ratio: {reward0_ratio}\nreward1_ratio: {reward1_ratio}\nreward2_ratio: {reward2_ratio}\nreward3_ratio: {reward3_ratio}')
        return reward_info
        

    def get_observation(self, isaac_envi):
        end_observation = isaac_envi.rb_states[isaac_envi.hand_idxs,].float()    # Left shape: (num_envi, 13). [0:3]: xyz, [3:7]: rotation quaternion, [7:10]: xyz velocity, [10:13]: rotation velocity.
        end_obs_mean, end_obs_std = self.stats['observations/end_observation_mean'].to(end_observation.device), self.stats['observations/end_observation_std'].to(end_observation.device)
        norm_end_observation = (end_observation - end_obs_mean) / end_obs_std # Left shape: (num_envi, 13)

        joint_observation = isaac_envi.dof_pos[:, :, 0].float()  # Left shape: (num_envi, 9)
        joint_obs_mean, joint_obs_std = self.stats['observations/joint_observation_mean'].to(joint_observation.device), self.stats['observations/joint_observation_std'].to(joint_observation.device)
        norm_joint_observation = (joint_observation - joint_obs_mean) / joint_obs_std # Left shape: (num_envi, 13)

        images = []
        for cnt, env in enumerate(isaac_envi.envs):
            env_image_dict = {}
            if 'top_camera' in self.cfg['DATA']['CAMERA_NAMES']:
                top_rgb_image = isaac_envi.gym.get_camera_image(isaac_envi.sim, env, isaac_envi.cameras[cnt]['top_camera'], gymapi.IMAGE_COLOR)
                top_rgb_image = top_rgb_image.reshape(240, 320, 4)[:, :, :3]
                env_image_dict['top_camera'] = top_rgb_image
            if 'exterior_camera1' in self.cfg['DATA']['CAMERA_NAMES']:
                side1_rgb_image = isaac_envi.gym.get_camera_image(isaac_envi.sim, env, isaac_envi.cameras[cnt]['side_camera1'], gymapi.IMAGE_COLOR)
                side1_rgb_image = side1_rgb_image.reshape(240, 320, 4)[:, :, :3]
                env_image_dict['exterior_camera1'] = side1_rgb_image
            if 'exterior_camera2' in self.cfg['DATA']['CAMERA_NAMES']:
                side2_rgb_image = isaac_envi.gym.get_camera_image(isaac_envi.sim, env, isaac_envi.cameras[cnt]['side_camera2'], gymapi.IMAGE_COLOR)
                side2_rgb_image = side2_rgb_image.reshape(240, 320, 4)[:, :, :3]
                env_image_dict['exterior_camera2'] = side2_rgb_image
            if 'wrist_camera' in self.cfg['DATA']['CAMERA_NAMES']:
                hand_rgb_image = isaac_envi.gym.get_camera_image(isaac_envi.sim, env, isaac_envi.cameras[cnt]['hand_camera'], gymapi.IMAGE_COLOR)
                hand_rgb_image = hand_rgb_image.reshape(240, 320, 4)[:, :, :3]
                env_image_dict['wrist_camera'] = hand_rgb_image
            images.append(env_image_dict)

        all_cam_image_list = []
        for camera_name in self.cfg['DATA']['CAMERA_NAMES']:
            single_camera_images = np.stack([ele[camera_name] for ele in images], axis = 0)   # Left shape: (num_envi, H, W, C)
            all_cam_image_list.append(single_camera_images)
        
        all_cam_images = np.stack(all_cam_image_list, axis = 0)   # Left shape: (num_camera, num_envi, H, W, C)
        all_cam_images = all_cam_images.transpose(1, 0, 4, 2, 3)     # Left shape: (num_envi, num_camera, C, H, W)
        num_envi, num_cam, C, H, W = all_cam_images.shape
        all_cam_images = torch.from_numpy(all_cam_images).float().reshape(num_envi * num_cam, C, H, W).cuda()
        all_cam_images = all_cam_images.float() / 255
        resize_w, resize_h = self.cfg['DATA']['IMG_RESIZE_SHAPE'][0], self.cfg['DATA']['IMG_RESIZE_SHAPE'][1]
        all_cam_images = F.resize(all_cam_images, size=(resize_h, resize_w))
        all_cam_images = F.normalize(all_cam_images, mean=self.cfg['DATA']['IMG_NORM_MEAN'], std=self.cfg['DATA']['IMG_NORM_STD'])  # Left shape: (num_envi * num_camera, C, H, W)
        all_cam_images = all_cam_images.view(num_envi, num_cam, C, resize_h, resize_w)
        
        return norm_end_observation, norm_joint_observation, all_cam_images
    
    def prepare_policy_input(self, end_obs, joint_obs, past_action, all_cam_images, task_instruction, status_pred):
        assert len(end_obs) == len(joint_obs)
        end_obs = torch.stack(end_obs, dim = 1) # Left shape: (num_env, T, 13)
        joint_obs = torch.stack(joint_obs, dim = 1) # Left shape: (num_env, T, 9)
        past_action = torch.stack(past_action, dim = 1) # Left shape: (num_env, T, 9)

        past_obs_len, obs_sample_interval = self.cfg['DATA']['PAST_OBSERVATION_LEN'], self.cfg['DATA']['OBSERVATION_SAMPLE_INTERVAL']
        if end_obs.shape[1] >= (past_obs_len - 1) * obs_sample_interval + 1:
            end_obs = end_obs[:, end_obs.shape[1] - (past_obs_len - 1) * obs_sample_interval - 1 : end_obs.shape[1] : obs_sample_interval]   # Left shape: (num_env, past_obs_len, 13)
            joint_obs = joint_obs[:, joint_obs.shape[1] - (past_obs_len - 1) * obs_sample_interval -1 : joint_obs.shape[1] : obs_sample_interval]   # Left shape: (num_env, past_obs_len, 9)
            observation_is_pad = torch.zeros((end_obs.shape[0], end_obs.shape[1]), dtype = torch.bool).to(end_obs.device)  # Left shape: (num_env, past_obs_len)
        else:
            valid_past_num = (end_obs.shape[1] - 1) // obs_sample_interval
            st = (end_obs.shape[1] - 1) - valid_past_num * obs_sample_interval
            end_obs = end_obs[:, st : end_obs.shape[1] : obs_sample_interval]   # Left shape: (num_env, past_obs_len, 13)
            joint_obs = joint_obs[:, st : joint_obs.shape[1] : obs_sample_interval]   # Left shape: (num_env, past_obs_len, 9)
            observation_is_pad = torch.zeros((end_obs.shape[0], end_obs.shape[1]), dtype = torch.bool).to(end_obs.device)
            observation_is_pad = torch.cat((torch.ones((end_obs.shape[0], past_obs_len - end_obs.shape[1]), dtype = torch.bool).to(end_obs.device), observation_is_pad), dim = 1)
            pad_end_obs = torch.zeros((end_obs.shape[0], past_obs_len - end_obs.shape[1], end_obs.shape[2]), dtype = torch.float32).to(end_obs.device)
            pad_joint_obs = torch.zeros((joint_obs.shape[0], past_obs_len - joint_obs.shape[1], joint_obs.shape[2]), dtype = torch.float32).to(end_obs.device)
            end_obs = torch.cat((pad_end_obs, end_obs), dim = 1)
            joint_obs = torch.cat((pad_joint_obs, joint_obs), dim = 1)

        past_action_len, past_action_interval = self.cfg['DATA']['PAST_ACTION_LEN'], self.cfg['DATA']['PAST_ACTION_SAMPLE_INTERVAL']
        if past_action.shape[1] >= (past_action_len - 1) * past_action_interval + 1:
            past_action = past_action[:, past_action.shape[1] - (past_action_len - 1) * past_action_interval - 1: past_action.shape[1] : past_action_interval]
            past_action_is_pad = torch.zeros((past_action.shape[0], past_action.shape[1]), dtype = torch.bool).to(past_action.device)  # Left shape: (num_env, T, 9)
        else:
            valid_past_num = (past_action.shape[1] - 1) // past_action_interval
            st = (past_action.shape[1] - 1) - valid_past_num * past_action_interval
            past_action = past_action[:, st : past_action.shape[1] : past_action_interval]   # Left shape: (num_env, past_action_len, 9)
            past_action_is_pad = torch.zeros((past_action.shape[0], past_action.shape[1]), dtype = torch.bool).to(past_action.device)
            past_action_is_pad = torch.cat((torch.ones((past_action.shape[0], past_action_len - past_action.shape[1]), dtype = torch.bool).to(past_action.device), past_action_is_pad), dim = 1)
            pad_past_action = torch.zeros((past_action.shape[0], past_action_len - past_action.shape[1], past_action.shape[2]), dtype = torch.float32).to(past_action.device)
            past_action = torch.cat((pad_past_action, past_action), dim = 1)

        return all_cam_images.cuda(), past_action.cuda(), end_obs.cuda(), joint_obs.cuda(), past_action_is_pad.cuda(), observation_is_pad.cuda(), task_instruction, status_pred
    
    def execute_action(self, action, isaac_envi):
        hand_pos = isaac_envi.rb_states[isaac_envi.hand_idxs, :3] # Left shape: (num_envi, 3)
        hand_rot = isaac_envi.rb_states[isaac_envi.hand_idxs, 3:7] # Left shape: (num_envi, 4)

        goal_pos = action[:, :3]    # Left shape: (num_envi, 3)
        goal_rot = action[:, 3:7]   # Left shape: (num_envi, 4)
        goal_gripper = action[:, 7:]    # Left shape: (num_envi, 2)

        pos_err = goal_pos - hand_pos
        orn_err = isaac_envi.orientation_error(goal_rot, hand_rot)
        dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)
        arm_ctrl = isaac_envi.dof_pos.squeeze(-1)[:, :7] + isaac_envi.control_ik(dpose)   # Control all joints except the gripper.

        isaac_envi.pos_action[:, :7] = arm_ctrl
        isaac_envi.pos_action[:, 7:] = goal_gripper

    def get_reward(self, isaac_envi):
        task_instruction_list = isaac_envi.task_instruction
        box1_idx_list, box2_idx_list = [], []
        for box_idx_dict, task_instruction in zip(isaac_envi.box_idxs, task_instruction_list):
            task_instruction = task_instruction.split(' ')
            box1_idx_list.append(box_idx_dict[task_instruction[3]])
            box2_idx_list.append(box_idx_dict[task_instruction[11]])

        box1_xyz = isaac_envi.rb_states[box1_idx_list, :3]  # Left shape: (num_env, 3)
        box2_xyz = isaac_envi.rb_states[box2_idx_list, :3]  # Left shape: (num_env, 3)

        reward = torch.zeros((box1_xyz.shape[0]), dtype = torch.float32).to(box1_xyz.device)
        reward1_mask = (box1_xyz[:, 0] > 0.39) & (box1_xyz[:, 0] < 0.61) & (box1_xyz[:, 1] > -0.26) & (box1_xyz[:, 1] < -0.14)
        reward[reward1_mask] += 1
        reward2_mask = reward1_mask & (box2_xyz[:, 0] > 0.39) & (box2_xyz[:, 0] < 0.61) & (box2_xyz[:, 1] > -0.26) & (box2_xyz[:, 1] < -0.14)
        reward[reward2_mask] += 1
        reward3_mask = reward2_mask & (torch.norm(box1_xyz[:, 0:2] - box2_xyz[:, 0:2], dim = 1) < 0.032) & (box2_xyz[:, 2] - box1_xyz[:, 2] > 0.022)
        reward[reward3_mask] += 1
        
        return reward