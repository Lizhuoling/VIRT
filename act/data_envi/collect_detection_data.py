import cv2
import pdb
import os
import time

class detection_data_collector():
    def __init__(self, save_data_root):
        self.save_data_path = save_data_root
        folders = ['usb', 'cam_high', 'cam_left_wrist', 'cam_right_wrist']
        for folder in folders:
            if not os.path.exists(os.path.join(self.save_data_path, folder)):
                os.makedirs(os.path.join(self.save_data_path, folder))

        if cam_mode == 'usb':
            self.cap = cv2.VideoCapture(0)
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
                break
            elif press_key == ord('b'): # begin recording
                print('Begin recording.')
                record_flag = True
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
                record_flag = False

            time.sleep(0.05)

        self.release_cap()

    def release_cap(self,):
        self.cap.release()

if __name__ == '__main__':
    save_data_root = '/home/cvte/twilight/home/data/aloha_beverage'
    cam_mode = 'usb'    # usb, cam_high, cam_left_wrist, or cam_right_wrist

    collector = detection_data_collector(save_data_root = save_data_root)
    collector.collect_data_main(cam_mode)