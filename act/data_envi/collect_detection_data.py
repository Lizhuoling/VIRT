import cv2
import pdb
import os
import time
import rospy
import numpy as np

from act.data_envi.collect_data_aloha import ros_default_args, RosOperator

class singlecamera_det_data_collector():
    def __init__(self, save_data_root, cam_mode):
        self.save_data_path = save_data_root
        folders = ['usb', 'cam_high', 'cam_left_wrist', 'cam_right_wrist']
        for folder in folders:
            if not os.path.exists(os.path.join(self.save_data_path, folder)):
                os.makedirs(os.path.join(self.save_data_path, folder))

        if cam_mode == 'usb':
            self.cap = cv2.VideoCapture(6)
        elif cam_mode == 'cam_high':
            self.cap = cv2.VideoCapture(2)
        elif cam_mode == 'cam_left_wrist':
            self.cap = cv2.VideoCapture(0)
        elif cam_mode == 'cam_right_wrist':
            self.cap = cv2.VideoCapture(4)
        else:
            raise NotImplementedError

    def collect_data_main(self, cam_mode):
        assert cam_mode in ['usb', 'cam_high', 'cam_left_wrist', 'cam_right_wrist']

        record_flag = False
        record_frames = []
        while True:
            ret, frame = self.cap.read()
            cv2.imshow('Camera Frame', frame)
            if record_flag:
                record_frames.append(frame)

            press_key = cv2.waitKey(1)
            if press_key == ord('q'):   # quit
                print('Quit.')
                record_frames = []
                break
            elif press_key == ord('b'): # begin recording
                print('Begin recording.')
                record_flag = True
                record_frames = []
            elif press_key == ord('r'): # restart recording
                print('Restart recording.')
                record_flag = False
                record_frames = []
            elif press_key == ord('s'):  # save
                if len(record_frames) == 0:
                    print("Please first record some frames and then save!")
                    continue
                print('Save the recorded video.')
                save_video_root = os.path.join(self.save_data_path, cam_mode)
                episode_cnt = len(os.listdir(save_video_root))
                save_video_path = os.path.join(save_video_root, 'episode_%d.mp4' % episode_cnt)
                fps = 20
                frame_size = (record_frames[0].shape[1], record_frames[0].shape[0])
                codec = cv2.VideoWriter_fourcc(*'mp4v')
                video_writter = cv2.VideoWriter(save_video_path, codec, fps, frame_size)
                for record_frame in record_frames:
                    video_writter.write(record_frame)
                video_writter.release()
                print("The video is saved at {}".format(save_video_path))
                record_frames = []
                record_flag = False

            time.sleep(0.05)

        self.release_cap()

    def release_cap(self,):
        self.cap.release()

class multicamera_det_data_collector():
    def __init__(self, save_data_root):
        self.save_data_path = save_data_root
        self.cam_names = ['cam_high', 'cam_left_wrist', 'cam_right_wrist']

        self.ros_args = ros_default_args()
        self.ros_args.publish_rate = 20
        self.save_data_count = 5
        self.ros_operator = RosOperator(self.ros_args)

    def collect_data_main(self):
        freq_controller = rospy.Rate(self.ros_args.publish_rate)

        record_flag = False
        record_frame_dict = {cam_name: [] for cam_name in self.cam_names}
        cnt = 0
        while True:
            result = self.ros_operator.get_frame()
            if not result:
                freq_controller.sleep()
                continue
            
            (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth, puppet_arm_left, puppet_arm_right, master_arm_left, master_arm_right, robot_base) = result   # rgb images

            img_dict = dict(cam_high = img_front[:, :, ::-1], cam_left_wrist = img_left[:, :, ::-1], cam_right_wrist = img_right[:, :, ::-1])
            vis_img = np.ascontiguousarray(np.concatenate((img_left[:, :, ::-1], img_front[:, :, ::-1], img_right[:, :, ::-1]), axis = 1))
            cv2.imshow('Camera Frame', vis_img)

            if record_flag and cnt % self.save_data_count == 0:
                for cam_name in self.cam_names:
                    record_frame_dict[cam_name].append(np.ascontiguousarray(img_dict[cam_name]))

            press_key = cv2.waitKey(1)
            if press_key == ord('q'):   # quit
                print('Quit.')
                record_frame_dict = {cam_name: [] for cam_name in self.cam_names}
                break
            elif press_key == ord('b'): # begin recording
                print('Begin recording.')
                record_flag = True
                record_frame_dict = {cam_name: [] for cam_name in self.cam_names}
            elif press_key == ord('r'): # restart recording
                print('Restart recording.')
                record_flag = False
                record_frame_dict = {cam_name: [] for cam_name in self.cam_names}
            elif press_key == ord('s'):  # save
                if len(record_frame_dict['cam_high']) == 0:
                    print("Please first record some frames and then save!")
                    continue
                print('Save the recorded frames.')
                batch_num = self.get_max_folder_cnt(self.save_data_path)
                save_frame_root = os.path.join(self.save_data_path, 'batch_{}'.format(batch_num))
                if not os.path.exists(save_frame_root): os.makedirs(save_frame_root)
                for cam_name in self.cam_names:
                    save_frame_folder = os.path.join(save_frame_root, cam_name)
                    if not os.path.exists(save_frame_folder):
                        os.makedirs(save_frame_folder)

                    #for cnt, record_frame in enumerate(record_frame_dict[cam_name]):
                    #    cv2.imwrite(os.path.join(save_frame_folder, f'frame_{cnt}.jpg'), record_frame)
                
                    fps = 20
                    frame_size = (record_frame_dict[cam_name][0].shape[1], record_frame_dict[cam_name][0].shape[0])
                    codec = cv2.VideoWriter_fourcc(*'mp4v')
                    video_cnt = len(os.listdir(save_frame_folder))
                    save_video_path = os.path.join(save_frame_folder, f'episode_{video_cnt}.mp4')
                    video_writter = cv2.VideoWriter(save_video_path, codec, fps, frame_size)
                    for record_frame in record_frame_dict[cam_name]:
                        video_writter.write(record_frame)
                    video_writter.release()
                    
                print('Saved.')
                record_frame_dict = {cam_name: [] for cam_name in self.cam_names}
                record_flag = False

            freq_controller.sleep()
            cnt += 1

    def get_max_folder_cnt(self, path):
        max_cnt = -1
        
        for folder_name in os.listdir(path):
            if '_' in folder_name:
                cnt = int(folder_name.split('_')[-1])
                if cnt > max_cnt:
                    max_cnt = cnt
        return max_cnt + 1

if __name__ == '__main__':
    save_data_root = '/home/agilex/twilight/data/aloha_beverage/aloha_beverage_multicam_detvideo'

    collector = multicamera_det_data_collector(save_data_root = save_data_root)
    collector.collect_data_main()