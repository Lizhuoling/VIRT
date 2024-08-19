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
import matplotlib.pyplot as plt
import threading
import rospy

from act.utils.inference.aloha_ros import RosOperator, ros_default_args

class AlohaReplayer():
    def __init__(self,):
        self.ros_args = ros_default_args()
        self.ros_operator = RosOperator(self.ros_args)

        self.init_moving_max_gap = 0.1

    def replay(self, h5py_path):
        h5f = h5py.File(h5py_path, 'r')
        actions = h5f['action'][:]  # Left shape: (T, 9)
        h5f.close()
        self.init_check()

        freq_controller = rospy.Rate(self.ros_args.publish_rate)
        action_cnt = 0
        while action_cnt < actions.shape[0]:
            qpos = self.get_observation()   # Left shape: (joint_dim,)
            action_to_execute = torch.Tensor(actions[action_cnt]) # Left shape: (joint_dim,)

            interp_flag = (action_to_execute - qpos).abs() > self.init_moving_max_gap
            interp_flag[[6, 13]] = False
            if interp_flag.sum() == 0:  # No interpolation is needed
                action_cnt += 1
            else:
                action_to_execute = torch.where(interp_flag, qpos + (action_to_execute - qpos).sign() * self.init_moving_max_gap, action_to_execute)
            left_action = action_to_execute[:7]
            right_action = action_to_execute[7:]
            self.ros_operator.puppet_arm_publish(left_action, right_action)
            freq_controller.sleep()

        print('Done!')

    def init_check(self,):
        # The robotic hands will nod to check whether they work properly.
        left0 = [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, 3.557830810546875]
        right0 = [-0.00133514404296875, 0.00438690185546875, 0.034523963928222656, -0.053597450256347656, -0.00476837158203125, -0.00209808349609375, 3.557830810546875]
        left1 = [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, -0.3393220901489258]
        right1 = [-0.00133514404296875, 0.00247955322265625, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, -0.3397035598754883]
        
        self.ros_operator.puppet_arm_publish_continuous(left0, right0)
        input("Enter any key to continue :")
        self.ros_operator.puppet_arm_publish_continuous(left1, right1)

    def get_observation(self,):
        freq_controller = rospy.Rate(self.ros_args.publish_rate)

        while True and not rospy.is_shutdown():
            result = self.ros_operator.get_frame()
            if not result:
                freq_controller.sleep()
                continue
            (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth, puppet_arm_left, puppet_arm_right, robot_base) = result  # The returned are rgb images.

            _, left_qpos, _ = puppet_arm_left.effort, puppet_arm_left.position, puppet_arm_left.velocity
            _, right_qpos, _ = puppet_arm_right.effort, puppet_arm_right.position, puppet_arm_right.velocity
            qpos = torch.Tensor(left_qpos + right_qpos)

            return qpos

if __name__ == '__main__':
    aloha_replayer = AlohaReplayer()
    aloha_replayer.replay(h5py_path = '/home/agilex/twilight/data/aloha_singleobj_grasp/h5py/episode_1.hdf5')