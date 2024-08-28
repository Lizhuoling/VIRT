import pdb
import os
import math
import numpy as np
import time
import cv2
import torch
import h5py
from torchvision.transforms import functional as F
import threading
import collections
from collections import deque
import argparse

import rospy
from std_msgs.msg import Header
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState, Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge

def ros_default_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--camera_names', action='store', type=str, help='camera_names',
                        default=['cam_high', 'cam_left_wrist', 'cam_right_wrist'], required=False)
    #  topic name of color image
    parser.add_argument('--img_front_topic', action='store', type=str, help='img_front_topic',
                        default='/camera_f/color/image_raw', required=False)
    parser.add_argument('--img_left_topic', action='store', type=str, help='img_left_topic',
                        default='/camera_l/color/image_raw', required=False)
    parser.add_argument('--img_right_topic', action='store', type=str, help='img_right_topic',
                        default='/camera_r/color/image_raw', required=False)
    
    # topic name of depth image
    parser.add_argument('--img_front_depth_topic', action='store', type=str, help='img_front_depth_topic',
                        default='/camera_f/depth/image_raw', required=False)
    parser.add_argument('--img_left_depth_topic', action='store', type=str, help='img_left_depth_topic',
                        default='/camera_l/depth/image_raw', required=False)
    parser.add_argument('--img_right_depth_topic', action='store', type=str, help='img_right_depth_topic',
                        default='/camera_r/depth/image_raw', required=False)
    
    # topic name of arm
    parser.add_argument('--master_arm_left_topic', action='store', type=str, help='master_arm_left_topic',
                        default='/master/joint_left', required=False)
    parser.add_argument('--master_arm_right_topic', action='store', type=str, help='master_arm_right_topic',
                        default='/master/joint_right', required=False)
    parser.add_argument('--puppet_arm_left_topic', action='store', type=str, help='puppet_arm_left_topic',
                        default='/puppet/joint_left', required=False)
    parser.add_argument('--puppet_arm_right_topic', action='store', type=str, help='puppet_arm_right_topic',
                        default='/puppet/joint_right', required=False)
    
    # topic name of robot_base
    parser.add_argument('--robot_base_topic', action='store', type=str, help='robot_base_topic',
                        default='/odom', required=False)
    
    parser.add_argument('--use_robot_base', action='store', type=bool, help='use_robot_base',
                        default=False, required=False)
    # collect depth image
    parser.add_argument('--use_depth_image', action='store', type=bool, help='use_depth_image',
                        default=False, required=False)
    
    parser.add_argument('--publish_rate', action='store', type=int, help='publish_rate',
                        default=40, required=False)
    
    args = parser.parse_args(args=[])
    return args

class RosOperator:
    def __init__(self, args):
        self.robot_base_deque = None
        self.puppet_arm_right_deque = None
        self.puppet_arm_left_deque = None
        self.master_arm_right_deque = None
        self.master_arm_left_deque = None
        self.img_front_deque = None
        self.img_right_deque = None
        self.img_left_deque = None
        self.img_front_depth_deque = None
        self.img_right_depth_deque = None
        self.img_left_depth_deque = None
        self.bridge = None
        self.args = args
        self.init()
        self.init_ros()

    def init(self):
        self.bridge = CvBridge()
        self.img_left_deque = deque()
        self.img_right_deque = deque()
        self.img_front_deque = deque()
        self.img_left_depth_deque = deque()
        self.img_right_depth_deque = deque()
        self.img_front_depth_deque = deque()
        self.master_arm_left_deque = deque()
        self.master_arm_right_deque = deque()
        self.puppet_arm_left_deque = deque()
        self.puppet_arm_right_deque = deque()
        self.robot_base_deque = deque()

    def get_frame(self):
        if len(self.img_left_deque) == 0 or len(self.img_right_deque) == 0 or len(self.img_front_deque) == 0 or \
                (self.args.use_depth_image and (len(self.img_left_depth_deque) == 0 or len(self.img_right_depth_deque) == 0 or len(self.img_front_depth_deque) == 0)):
            return False
        if self.args.use_depth_image:
            frame_time = min([self.img_left_deque[-1].header.stamp.to_sec(), self.img_right_deque[-1].header.stamp.to_sec(), self.img_front_deque[-1].header.stamp.to_sec(),
                              self.img_left_depth_deque[-1].header.stamp.to_sec(), self.img_right_depth_deque[-1].header.stamp.to_sec(), self.img_front_depth_deque[-1].header.stamp.to_sec()])
        else:
            frame_time = min([self.img_left_deque[-1].header.stamp.to_sec(), self.img_right_deque[-1].header.stamp.to_sec(), self.img_front_deque[-1].header.stamp.to_sec()])

        if len(self.img_left_deque) == 0 or self.img_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.img_right_deque) == 0 or self.img_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.img_front_deque) == 0 or self.img_front_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.master_arm_left_deque) == 0 or self.master_arm_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.master_arm_right_deque) == 0 or self.master_arm_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.puppet_arm_left_deque) == 0 or self.puppet_arm_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.puppet_arm_right_deque) == 0 or self.puppet_arm_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if self.args.use_depth_image and (len(self.img_left_depth_deque) == 0 or self.img_left_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if self.args.use_depth_image and (len(self.img_right_depth_deque) == 0 or self.img_right_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if self.args.use_depth_image and (len(self.img_front_depth_deque) == 0 or self.img_front_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if self.args.use_robot_base and (len(self.robot_base_deque) == 0 or self.robot_base_deque[-1].header.stamp.to_sec() < frame_time):
            return False

        while self.img_left_deque[0].header.stamp.to_sec() < frame_time:
            self.img_left_deque.popleft()
        img_left = self.bridge.imgmsg_to_cv2(self.img_left_deque.popleft(), 'passthrough')
        # print("img_left:", img_left.shape)

        while self.img_right_deque[0].header.stamp.to_sec() < frame_time:
            self.img_right_deque.popleft()
        img_right = self.bridge.imgmsg_to_cv2(self.img_right_deque.popleft(), 'passthrough')

        while self.img_front_deque[0].header.stamp.to_sec() < frame_time:
            self.img_front_deque.popleft()
        img_front = self.bridge.imgmsg_to_cv2(self.img_front_deque.popleft(), 'passthrough')

        while self.master_arm_left_deque[0].header.stamp.to_sec() < frame_time:
            self.master_arm_left_deque.popleft()
        master_arm_left = self.master_arm_left_deque.popleft()

        while self.master_arm_right_deque[0].header.stamp.to_sec() < frame_time:
            self.master_arm_right_deque.popleft()
        master_arm_right = self.master_arm_right_deque.popleft()

        while self.puppet_arm_left_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_left_deque.popleft()
        puppet_arm_left = self.puppet_arm_left_deque.popleft()

        while self.puppet_arm_right_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_right_deque.popleft()
        puppet_arm_right = self.puppet_arm_right_deque.popleft()

        img_left_depth = None
        if self.args.use_depth_image:
            while self.img_left_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_left_depth_deque.popleft()
            img_left_depth = self.bridge.imgmsg_to_cv2(self.img_left_depth_deque.popleft(), 'passthrough')
            top, bottom, left, right = 40, 40, 0, 0
            img_left_depth = cv2.copyMakeBorder(img_left_depth, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

        img_right_depth = None
        if self.args.use_depth_image:
            while self.img_right_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_right_depth_deque.popleft()
            img_right_depth = self.bridge.imgmsg_to_cv2(self.img_right_depth_deque.popleft(), 'passthrough')
        top, bottom, left, right = 40, 40, 0, 0
        img_right_depth = cv2.copyMakeBorder(img_right_depth, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

        img_front_depth = None
        if self.args.use_depth_image:
            while self.img_front_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_front_depth_deque.popleft()
            img_front_depth = self.bridge.imgmsg_to_cv2(self.img_front_depth_deque.popleft(), 'passthrough')
        top, bottom, left, right = 40, 40, 0, 0
        img_front_depth = cv2.copyMakeBorder(img_front_depth, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

        robot_base = None
        if self.args.use_robot_base:
            while self.robot_base_deque[0].header.stamp.to_sec() < frame_time:
                self.robot_base_deque.popleft()
            robot_base = self.robot_base_deque.popleft()

        return (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
                puppet_arm_left, puppet_arm_right, master_arm_left, master_arm_right, robot_base)

    def img_left_callback(self, msg):
        if len(self.img_left_deque) >= 2000:
            self.img_left_deque.popleft()
        self.img_left_deque.append(msg)

    def img_right_callback(self, msg):
        if len(self.img_right_deque) >= 2000:
            self.img_right_deque.popleft()
        self.img_right_deque.append(msg)

    def img_front_callback(self, msg):
        if len(self.img_front_deque) >= 2000:
            self.img_front_deque.popleft()
        self.img_front_deque.append(msg)

    def img_left_depth_callback(self, msg):
        if len(self.img_left_depth_deque) >= 2000:
            self.img_left_depth_deque.popleft()
        self.img_left_depth_deque.append(msg)

    def img_right_depth_callback(self, msg):
        if len(self.img_right_depth_deque) >= 2000:
            self.img_right_depth_deque.popleft()
        self.img_right_depth_deque.append(msg)

    def img_front_depth_callback(self, msg):
        if len(self.img_front_depth_deque) >= 2000:
            self.img_front_depth_deque.popleft()
        self.img_front_depth_deque.append(msg)

    def master_arm_left_callback(self, msg):
        if len(self.master_arm_left_deque) >= 2000:
            self.master_arm_left_deque.popleft()
        self.master_arm_left_deque.append(msg)

    def master_arm_right_callback(self, msg):
        if len(self.master_arm_right_deque) >= 2000:
            self.master_arm_right_deque.popleft()
        self.master_arm_right_deque.append(msg)

    def puppet_arm_left_callback(self, msg):
        if len(self.puppet_arm_left_deque) >= 2000:
            self.puppet_arm_left_deque.popleft()
        self.puppet_arm_left_deque.append(msg)

    def puppet_arm_right_callback(self, msg):
        if len(self.puppet_arm_right_deque) >= 2000:
            self.puppet_arm_right_deque.popleft()
        self.puppet_arm_right_deque.append(msg)

    def robot_base_callback(self, msg):
        if len(self.robot_base_deque) >= 2000:
            self.robot_base_deque.popleft()
        self.robot_base_deque.append(msg)

    def init_ros(self):
        rospy.init_node('record_episodes', anonymous=True)
        rospy.Subscriber(self.args.img_left_topic, Image, self.img_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_right_topic, Image, self.img_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_front_topic, Image, self.img_front_callback, queue_size=1000, tcp_nodelay=True)
        if self.args.use_depth_image:
            rospy.Subscriber(self.args.img_left_depth_topic, Image, self.img_left_depth_callback, queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(self.args.img_right_depth_topic, Image, self.img_right_depth_callback, queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(self.args.img_front_depth_topic, Image, self.img_front_depth_callback, queue_size=1000, tcp_nodelay=True)
        
        rospy.Subscriber(self.args.master_arm_left_topic, JointState, self.master_arm_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.master_arm_right_topic, JointState, self.master_arm_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.puppet_arm_left_topic, JointState, self.puppet_arm_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.puppet_arm_right_topic, JointState, self.puppet_arm_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.robot_base_topic, Odometry, self.robot_base_callback, queue_size=1000, tcp_nodelay=True)

class SaveDataManager():
    def __init__(self, data_root, start_idx = None):
        self.data_root = data_root
        assert os.path.exists(self.data_root), "The path {} does not exist.".format(self.data_root)
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
        self.cam_high_path = os.path.join(self.data_root, 'cam_high')
        self.cam_left_wrist_path  = os.path.join(self.data_root, 'cam_left_wrist')
        self.cam_right_wrist_path  = os.path.join(self.data_root, 'cam_right_wrist')
        if not os.path.exists(self.h5py_path): os.makedirs(self.h5py_path)
        if not os.path.exists(self.cam_high_path): os.makedirs(self.cam_high_path)
        if not os.path.exists(self.cam_left_wrist_path): os.makedirs(self.cam_left_wrist_path)
        if not os.path.exists(self.cam_right_wrist_path): os.makedirs(self.cam_right_wrist_path)

    def set_episode_index(self, index):
        self.episode_index = index

    def save_data(self, action_list, obs_list, images_list, depth_images_list, base_vel_list, task_instruction):
        assert len(action_list) == len(obs_list) == len(images_list) != 0
        img_height, img_width, _ = images_list[0]['cam_high'].shape
        
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        cam_high_writter = cv2.VideoWriter(os.path.join(self.cam_high_path, 'episode_{}.mp4'.format(self.episode_index)), fourcc, 20, (img_width, img_height))
        cam_left_wrist_writter = cv2.VideoWriter(os.path.join(self.cam_left_wrist_path, 'episode_{}.mp4'.format(self.episode_index)), fourcc, 20, (img_width, img_height))
        cam_right_wrist_writter = cv2.VideoWriter(os.path.join(self.cam_right_wrist_path, 'episode_{}.mp4'.format(self.episode_index)), fourcc, 20, (img_width, img_height))

        h5f = h5py.File(os.path.join(self.h5py_path, 'episode_{}.hdf5'.format(self.episode_index)), 'w')
        h5f['task_instruction'] = task_instruction
        effort_obs = np.array([ele['effort'] for ele in obs_list])
        qpos_obs = np.array([ele['qpos'] for ele in obs_list])
        qvel_obs = np.array([ele['qvel'] for ele in obs_list])
        h5f_observations = h5f.create_group('observations')
        h5f_observations['effort_obs'] = effort_obs
        h5f_observations['qpos_obs'] = qpos_obs
        h5f_observations['qvel_obs'] = qvel_obs
        h5f['action'] = np.array(action_list)
        if len(base_vel_list) != 0:
            base_vel = np.array([ele for ele in base_vel_list])
            h5f['base_vel'] = base_vel
        h5f.close()

        camera_keys = ['cam_high', 'cam_left_wrist', 'cam_right_wrist',]
        for image_dict in images_list:
            for key in camera_keys:
                if key == 'cam_high':
                    cam_high_writter.write(image_dict[key])
                elif key == 'cam_left_wrist':
                    cam_left_wrist_writter.write(image_dict[key])
                elif key == 'cam_right_wrist':
                    cam_right_wrist_writter.write(image_dict[key])

        cam_high_writter.release()
        cam_left_wrist_writter.release()
        cam_right_wrist_writter.release()

        self.episode_index += 1

def collect_data_main(task_name, save_data_path = "", total_episodes = 1):
    ros_args = ros_default_args()
    ros_operator = RosOperator(ros_args)
    freq_controller = rospy.Rate(ros_args.publish_rate)

    save_data_flag = False
    if save_data_path != "":
        save_data_flag = True

    if save_data_flag:
        save_data_manager = SaveDataManager(save_data_path)
        start_episode_num = save_data_manager.episode_index
    else:
        start_episode_num = 0

    task_instruction = 'Please make a cup of beverage by mixing the provided blueberry and mango juice using the juicer.'
    record_data_flag = False
    episode_cnt = start_episode_num
    while episode_cnt < total_episodes:
        print(f"Data collection episode {episode_cnt}/{total_episodes}...")
        while True:
            result = ros_operator.get_frame()
            if not result:
                freq_controller.sleep()
                continue
            
            (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth, puppet_arm_left, puppet_arm_right, master_arm_left, master_arm_right, robot_base) = result   # rgb images
            img_dict = dict(
                cam_high = img_front[:, :, ::-1],   # rgb to bgr
                cam_left_wrist = img_left[:, :, ::-1],  # rgb to bgr
                cam_right_wrist = img_right[:, :, ::-1],    # rgb to bgr
            )
            if ros_args.use_depth_image:
                img_depth_dict = dict(
                    cam_high_depth = img_front_depth,
                    cam_left_wrist_depth = img_left_depth,
                    cam_right_wrist_depth = img_right_depth,
                )
            qpos = np.concatenate((np.array(puppet_arm_left.position), np.array(puppet_arm_right.position)), axis=0)
            qvel = np.concatenate((np.array(puppet_arm_left.velocity), np.array(puppet_arm_right.velocity)), axis=0)
            effort = np.concatenate((np.array(puppet_arm_left.effort), np.array(puppet_arm_right.effort)), axis=0)
            obs_dict = dict(
                effort = effort,
                qpos = qpos,
                qvel = qvel,
            )
            if ros_args.use_robot_base:
                base_vel = [robot_base.twist.twist.linear.x, robot_base.twist.twist.angular.z]
                
            action = np.concatenate((np.array(master_arm_left.position), np.array(master_arm_right.position)), axis=0)

            if save_data_flag and record_data_flag:
                action_list.append(action)
                obs_list.append(obs_dict)
                images_list.append(img_dict)
                if ros_args.use_depth_image: depth_images_list.append(img_depth_dict)
                if ros_args.use_robot_base: base_vel_list.append(base_vel)

            vis_img = np.ascontiguousarray(np.concatenate((img_left[:, :, ::-1], img_front[:, :, ::-1], img_right[:, :, ::-1]), axis = 1))
            cv2.imshow('camera views', vis_img)
            key = cv2.waitKey(1)
            if key == ord('b'):
                print('Begin collect data')
                record_data_flag = True
                action_list, obs_list, images_list, depth_images_list, base_vel_list = [], [], [], [], []
            elif key == ord('r'): # Relay this episode without save 
                print('Replay')
                record_data_flag = False
                break
            elif key == ord('s'):
                print("Save data")
                save_data_manager.save_data(action_list, obs_list, images_list, depth_images_list, base_vel_list, task_instruction)
                episode_cnt += 1
                record_data_flag = False
                break
            elif key == ord('q'):
                print('Quit')
                return


if __name__ == '__main__':
    task_name = 'aloha_beverage'
    save_data_path = '/home/agilex/twilight/data/aloha_beverage/aloha_beverage'

    collect_data_main(task_name = task_name, save_data_path = save_data_path, total_episodes = 50)