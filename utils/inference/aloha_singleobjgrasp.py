import pdb
import math
import numpy as np
import time
import cv2
import torch
from torchvision.transforms import functional as F
import threading
import rospy
from std_msgs.msg import Header
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState, Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge

from utils.inference.aloha_ros import RosOperator, ros_default_args

class AlohaSingleObjGraspTestEnviManager():
    def __init__(self, cfg, policy, stats):
        self.cfg = cfg
        self.policy = policy
        self.stats = stats
        self.ros_args = ros_default_args()
        self.ros_operator = RosOperator(self.ros_args)

        self.inference_thread = None
        self.inference_lock = threading.Lock()
        self.inference_actions = None
        self.inference_timestep = None

        self.init_moving_max_gap = 0.1

    def init_check(self,):
        # The robotic hands will nod to check whether they work properly.
        left0 = [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, 3.557830810546875]
        right0 = [-0.00133514404296875, 0.00438690185546875, 0.034523963928222656, -0.053597450256347656, -0.00476837158203125, -0.00209808349609375, 3.557830810546875]
        left1 = [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, -0.3393220901489258]
        right1 = [-0.00133514404296875, 0.00247955322265625, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, -0.3397035598754883]
        
        self.ros_operator.puppet_arm_publish_continuous(left0, right0)
        input("Enter any key to continue :")
        self.ros_operator.puppet_arm_publish_continuous(left1, right1)
    
    def inference(self,):
        self.init_check()
        
        freq_controller = rospy.Rate(self.ros_args.publish_rate)
        with torch.inference_mode():
            cur_status = torch.zeros((1,), dtype = torch.long).cuda()
            if self.cfg['TASK_NAME'] == 'aloha_singleobjgrasp':
                task_instruction = 'Put the airphone box into the bin.'
            else:
                raise NotImplementedError
            effort_obs_list = []
            qpos_obs_list = []
            qvel_obs_list = []
            latest_qpos_unnorm = None
            action_list = []
            actions_pred = None
            action_step, action_cnt = 0, 0 # action_step is the total actions number that have been executed, and action_cnt is the number in the local action sequence.
            
            while action_step < self.cfg['EVAL']['INFERENCE_MAX_STEPS'] and not rospy.is_shutdown():
                # When no action has been predicted or all actions have been executed, the policy predicts new actions.
                if actions_pred == None or action_cnt >= actions_pred.shape[1]:
                    if len(action_list) == 0:   # This part will only be executed in the beginning of the inference.
                        norm_effort, norm_qpos, norm_qvel, imgs, latest_qpos_unnorm = self.get_observation()
                        effort_obs_list.append(norm_effort)
                        qpos_obs_list.append(norm_qpos)
                        qvel_obs_list.append(norm_qvel)
                        action_list.append(norm_qpos) # Initialize the first element with qpos observation
                    image, past_action, effort_obs, qpos_obs, qvel_obs, observation_is_pad, past_action_is_pad, task_instruction, status = self.prepare_policy_input(effort_obs_list, qpos_obs_list, \
                                    qvel_obs_list, action_list, imgs, task_instruction, cur_status)
                    norm_actions_pred, status_pred = self.policy(image = image, past_action = past_action, action = None, effort_obs = effort_obs, qpos_obs = qpos_obs, qvel_obs = qvel_obs, \
                                    observation_is_pad = observation_is_pad, past_action_is_pad = past_action_is_pad, action_is_pad = None, task_instruction = task_instruction, status = status)  # Left shape: (1, T, 9)
                    action_mean, action_std = self.stats['action_mean'][None, None].to(image.device), self.stats['action_std'][None, None].to(image.device) # Left shape: (1, 1, action_dim), (1, 1, action_dim)
                    actions_pred = norm_actions_pred * action_std + action_mean
                    #interp_actions_pred = self.interpolate_action(actions_pred = actions_pred, cur_qpos_obs = latest_qpos_unnorm)  # To make sure the beginning of this action sequence is executed smoothly.
                    #norm_interp_action_pred = (interp_actions_pred - action_mean) / action_std  # Left shape: (1, interp_num + chunk_size, joint_dim)
                    action_cnt = 0
                    cur_status = status_pred.clone()

                # Get and save observation data
                norm_effort, norm_qpos, norm_qvel, imgs, latest_qpos_unnorm = self.get_observation()
                effort_obs_list.append(norm_effort)
                qpos_obs_list.append(norm_qpos)
                qvel_obs_list.append(norm_qvel)
                
                # Execute an action
                action_to_execute = actions_pred[0, action_cnt].clone() # Left shape: (joint_dim,)
                interp_flag = (actions_pred[0, action_cnt] - latest_qpos_unnorm).abs() > self.init_moving_max_gap
                if interp_flag.sum() == 0:  # No interpolation is needed
                    action_cnt += 1
                else:
                    action_to_execute = torch.where(interp_flag, latest_qpos_unnorm + (actions_pred[0, action_cnt] - latest_qpos_unnorm).sign() * self.init_moving_max_gap, action_to_execute)
                norm_action_to_execute = (action_to_execute - self.stats['action_mean'].cuda()) / self.stats['action_std'].cuda()   # Left shape: (joint_dim,)
                action_list.append(norm_action_to_execute[None])
                left_action = action_to_execute[:7]
                right_action = action_to_execute[7:]
                self.ros_operator.puppet_arm_publish(left_action, right_action)
                action_step += 1

                freq_controller.sleep()

        return None
                    

    def get_observation(self,):
        freq_controller = rospy.Rate(self.ros_args.publish_rate)

        while True and not rospy.is_shutdown():
            result = self.ros_operator.get_frame()
            if not result:
                freq_controller.sleep()
                continue
            (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth, puppet_arm_left, puppet_arm_right, robot_base) = result  # The returned are rgb images.

            left_effort, left_qpos, left_qvel = puppet_arm_left.effort, puppet_arm_left.position, puppet_arm_left.velocity
            right_effort, right_qpos, right_qvel = puppet_arm_right.effort, puppet_arm_right.position, puppet_arm_right.velocity
            effort, qpos, qvel = torch.Tensor(left_effort + right_effort).cuda(), torch.Tensor(left_qpos + right_qpos).cuda(), torch.Tensor(left_qvel + right_qvel).cuda()
            norm_effort = ((effort - self.stats['observations/effort_mean'].cuda()) / self.stats['observations/effort_std'].cuda())[None]   # Left shape: (1, joint_dim)
            norm_qpos = ((qpos - self.stats['observations/qpos_mean'].cuda()) / self.stats['observations/qpos_std'].cuda())[None]   # Left shape: (1, joint_dim)
            norm_qvel = ((qvel - self.stats['observations/qvel_mean'].cuda()) / self.stats['observations/qvel_std'].cuda())[None]   # Left shape: (1, joint_dim)

            img_dict = dict(
                cam_high = torch.Tensor(img_front).cuda().permute(2, 0, 1),
                cam_left_wrist = torch.Tensor(img_left).cuda().permute(2, 0, 1), 
                cam_right_wrist = torch.Tensor(img_right).cuda().permute(2, 0, 1),
            )
            imgs = []
            for cam_name in self.cfg['DATA']['CAMERA_NAMES']:
                imgs.append(img_dict[cam_name])
            imgs = torch.stack(imgs, dim = 0)   # Left shape: (N, C, H, W)
            imgs = imgs / 255
            resize_w, resize_h = self.cfg['DATA']['IMG_RESIZE_SHAPE'][0], self.cfg['DATA']['IMG_RESIZE_SHAPE'][1]
            imgs = F.resize(imgs, size=(resize_h, resize_w))
            imgs = F.normalize(imgs, mean=self.cfg['DATA']['IMG_NORM_MEAN'], std=self.cfg['DATA']['IMG_NORM_STD'])[None]  # Left shape: (1, N, C, H, W)

            return norm_effort, norm_qpos, norm_qvel, imgs, qpos
        
    def prepare_policy_input(self, effort_obs_list, qpos_obs_list, qvel_obs_list, action_list, imgs, task_instruction, cur_status):
        effort_obs = torch.stack(effort_obs_list, dim = 1) # Left shape: (1, T, 14)
        qpos_obs = torch.stack(qpos_obs_list, dim = 1) # Left shape: (1, T, 14)
        qvel_obs = torch.stack(qvel_obs_list, dim = 1) # Left shape: (1, T, 14)
        past_action = torch.stack(action_list, dim = 1) # Left shape: (1, T, 14)

        past_obs_len, obs_sample_interval = self.cfg['DATA']['PAST_OBSERVATION_LEN'], self.cfg['DATA']['OBSERVATION_SAMPLE_INTERVAL']
        if effort_obs.shape[1] >= (past_obs_len - 1) * obs_sample_interval + 1:
            effort_obs = effort_obs[:, effort_obs.shape[1] - (past_obs_len - 1) * obs_sample_interval - 1 : effort_obs.shape[1] : obs_sample_interval]   # Left shape: (1, past_obs_len, 14)
            qpos_obs = qpos_obs[:, qpos_obs.shape[1] - (past_obs_len - 1) * obs_sample_interval -1 : qpos_obs.shape[1] : obs_sample_interval]   # Left shape: (1, past_obs_len, 14)
            qvel_obs = qvel_obs[:, qvel_obs.shape[1] - (past_obs_len - 1) * obs_sample_interval -1 : qvel_obs.shape[1] : obs_sample_interval]   # Left shape: (1, past_obs_len, 14)
            observation_is_pad = torch.zeros((effort_obs.shape[0], effort_obs.shape[1]), dtype = torch.bool).to(effort_obs.device)  # Left shape: (1, past_obs_len)
        else:
            valid_past_num = (effort_obs.shape[1] - 1) // obs_sample_interval
            st = (effort_obs.shape[1] - 1) - valid_past_num * obs_sample_interval
            effort_obs = effort_obs[:, st : effort_obs.shape[1] : obs_sample_interval]   # Left shape: (1, past_obs_len, 14)
            qpos_obs = qpos_obs[:, st : qpos_obs.shape[1] : obs_sample_interval]   # Left shape: (1, past_obs_len, 14)
            qvel_obs = qvel_obs[:, st : qvel_obs.shape[1] : obs_sample_interval]   # Left shape: (1, past_obs_len, 14)
            observation_is_pad = torch.zeros((effort_obs.shape[0], effort_obs.shape[1]), dtype = torch.bool).to(effort_obs.device)
            observation_is_pad = torch.cat((torch.ones((effort_obs.shape[0], past_obs_len - effort_obs.shape[1]), dtype = torch.bool).to(effort_obs.device), observation_is_pad), dim = 1)
            pad_effort_obs = torch.zeros((effort_obs.shape[0], past_obs_len - effort_obs.shape[1], effort_obs.shape[2]), dtype = torch.float32).to(effort_obs.device)
            effort_obs = torch.cat((pad_effort_obs, effort_obs), dim = 1)
            pad_qpos_obs = torch.zeros((qpos_obs.shape[0], past_obs_len - qpos_obs.shape[1], qpos_obs.shape[2]), dtype = torch.float32).to(effort_obs.device)
            qpos_obs = torch.cat((pad_qpos_obs, qpos_obs), dim = 1)
            pad_qvel_obs = torch.zeros((qvel_obs.shape[0], past_obs_len - qvel_obs.shape[1], qvel_obs.shape[2]), dtype = torch.float32).to(effort_obs.device)
            qvel_obs = torch.cat((pad_qvel_obs, qvel_obs), dim = 1)

        past_action_len, past_action_interval = self.cfg['DATA']['PAST_ACTION_LEN'], self.cfg['DATA']['PAST_ACTION_SAMPLE_INTERVAL']
        if past_action.shape[1] >= (past_action_len - 1) * past_action_interval + 1:
            past_action = past_action[:, past_action.shape[1] - (past_action_len - 1) * past_action_interval - 1: past_action.shape[1] : past_action_interval]
            past_action_is_pad = torch.zeros((past_action.shape[0], past_action.shape[1]), dtype = torch.bool).to(past_action.device)  # Left shape: (1, T, 14)
        else:
            valid_past_num = (past_action.shape[1] - 1) // past_action_interval
            st = (past_action.shape[1] - 1) - valid_past_num * past_action_interval
            past_action = past_action[:, st : past_action.shape[1] : past_action_interval]   # Left shape: (1, past_action_len, 14)
            past_action_is_pad = torch.zeros((past_action.shape[0], past_action.shape[1]), dtype = torch.bool).to(past_action.device)
            past_action_is_pad = torch.cat((torch.ones((past_action.shape[0], past_action_len - past_action.shape[1]), dtype = torch.bool).to(past_action.device), past_action_is_pad), dim = 1)
            pad_past_action = torch.zeros((past_action.shape[0], past_action_len - past_action.shape[1], past_action.shape[2]), dtype = torch.float32).to(past_action.device)
            past_action = torch.cat((pad_past_action, past_action), dim = 1)
        
        return imgs, past_action, effort_obs, qpos_obs, qvel_obs, observation_is_pad, past_action_is_pad, task_instruction, cur_status
    
    def interpolate_action(self, actions_pred, cur_qpos_obs):
        '''
            Input:
                actions_pred: The predicted action sequence. shape: (1, chunk_size, joint_dim).
                cur_qpos_obs: The latest qpos observation. shape: (joint_dim)
        '''
        init_actions_pred = actions_pred[0, 0].clone()  # Left shape: (joint_dim,)
        interp_step_num = ((init_actions_pred - cur_qpos_obs).abs().max() // self.init_moving_max_gap).item()
        interp_ratios = torch.linspace(0, 1, int(interp_step_num) + 1).cuda()[:-1]    # Left shape: (interpolation_num,)

        joint_dim = actions_pred.shape[-1]
        interpolation_num = interp_ratios.shape[0]
        interp_action = torch.lerp(cur_qpos_obs[None].expand(interpolation_num, -1), init_actions_pred[None].expand(interpolation_num, -1), interp_ratios[:, None].expand(-1, joint_dim)) # Left shape: (interp_num, joint_dim)
        
        interp_actions_pred = torch.cat((interp_action[None], actions_pred), dim = 1)   # Left shape: (1, interp_num + chunk_size, joint_dim)
        return interp_actions_pred