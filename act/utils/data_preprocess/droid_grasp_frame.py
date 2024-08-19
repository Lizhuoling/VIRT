import cv2
import numpy as np
import os
import pdb
import h5py
import tqdm

def droid_grasp_frame(droid_root, start_idx = 0):
    grasp_threshold = 0.5
    #close_duration_threshold = 10

    h5py_root = os.path.join(droid_root, 'h5py')
    exterior1_root = os.path.join(droid_root, 'exterior_image_1_left')
    exterior2_root = os.path.join(droid_root, 'exterior_image_2_left')
    wrist_root = os.path.join(droid_root, 'wrist_image_left')
    info_root = os.path.join(droid_root, 'info')

    h5py_filenames = sorted(os.listdir(h5py_root))
    for h5py_cnt in tqdm.tqdm(range(start_idx, len(h5py_filenames)), total = len(h5py_filenames), initial = start_idx):
        h5py_filename = h5py_filenames[h5py_cnt]
        h5py_path = os.path.join(h5py_root, h5py_filename)
        with h5py.File(h5py_path, 'r') as h5py_file:
            gripper_action = h5py_file['action_dict']['gripper_position'][:][:, 0]  # Left shape: (T,)
            cross_thre_moments = np.where((gripper_action[:-1] < grasp_threshold) & (gripper_action[1:] >= grasp_threshold))
            if cross_thre_moments[0].shape[0] == 0:
                continue
            cross_thre_moments = [ele[0] + 1 for ele in cross_thre_moments]   # Left shape: (cross_T,)
            
            '''close_flag = gripper_action >= grasp_threshold  # Left shape: (T,)
            close_cumsum = np.cumsum(close_flag[::-1])[::-1]    # Left shape: (T,)
            valid_cross_moments = []
            for i, cross_thre_moment in enumerate(cross_thre_moments):
                if i + 1 < len(cross_thre_moments):
                    close_duration = close_cumsum[i] - close_cumsum[i + 1]
                else:
                    close_duration = close_cumsum[i]
                if close_duration > close_duration_threshold: valid_cross_moments.append(cross_thre_moment)'''
        valid_cross_moments = cross_thre_moments

        with h5py.File(os.path.join(info_root, h5py_filename), 'w') as h5py_file:
            exterior1_frames = []
            exterior2_frames = []
            wrist_frames = []
            for valid_cross_moment in valid_cross_moments:
                video_filename = h5py_filename.replace('.hdf5', '.mp4')
                exterior1_frame = load_video_frame(os.path.join(exterior1_root, video_filename), valid_cross_moment)
                exterior2_frame = load_video_frame(os.path.join(exterior2_root, video_filename), valid_cross_moment)
                wrist_frame = load_video_frame(os.path.join(wrist_root, video_filename), valid_cross_moment)
                exterior1_frames.append(exterior1_frame)
                exterior2_frames.append(exterior2_frame)
                wrist_frames.append(wrist_frame)
            exterior1_frames = np.stack(exterior1_frames, axis = 0)
            exterior2_frames = np.stack(exterior2_frames, axis = 0)
            wrist_frames = np.stack(wrist_frames, axis = 0)
            h5py_file['exterior1_grasp_frames'] = exterior1_frames
            h5py_file['exterior2_grasp_frames'] = exterior2_frames
            h5py_file['wrist_grasp_frames'] = wrist_frames
            h5py_file['grasp_moments'] = np.array(cross_thre_moments)

    print('Done!')

        
def load_video_frame(video_path, frame_id):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except:
        print("frame: {}, video_path: {}, frame_id: {}".format(frame, video_path, frame_id))
        raise Exception("Error in cvtColor.")
    cap.release()
    return frame_rgb

click_point = None
def mouse_callback(event, x, y, flags, param):
    global click_point
    if event == cv2.EVENT_LBUTTONDOWN:
        click_point = (x, y)
        print("Update clock_point as: {}".format(click_point))

def annotate_graspframe_gripper(droid_root, start_idx = 0):
    info_root = os.path.join(droid_root, 'info')
    info_filenames = sorted(os.listdir(info_root))
    vis_shape = (1280, 640)
    global click_point

    for info_cnt in range(start_idx, len(info_filenames)):
        info_filename = info_filenames[info_cnt]
        info_path = os.path.join(info_root, info_filename)
        skip_flag = False
        with h5py.File(info_path, 'r+') as h5py_file:
            frame_keys = ['exterior1_grasp_frames', 'exterior2_grasp_frames',]
            for frame_key in frame_keys:
                print("info_name: {}, frame_key: {}, info_cnt: {}, info_total_num: {}".format(info_filename, frame_key, info_cnt, len(info_filenames)))
                frames = h5py_file[frame_key][:]
                frame_keypoint_list = []
                for frame_cnt in range(frames.shape[0]):
                    frame = frames[frame_cnt]
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    click_point = None
                    frame_h, frame_w, _ = frame.shape
                    vis_frame = cv2.resize(frame, vis_shape)
                    cv2.namedWindow('Key Frame Annotation')
                    cv2.setMouseCallback('Key Frame Annotation', mouse_callback)
                    cv2.imshow('Key Frame Annotation', vis_frame)
                    while True:
                        press_key = cv2.waitKey(100)
                        if press_key == ord('c'):
                            print('Complete one frame annotation.')
                            break
                        elif press_key == ord('s'):
                            print('Skip this sample.')
                            exit(0)
                    if click_point != None:
                        click_point = (int(click_point[0] * frame_w / vis_shape[0]), int(click_point[1] * frame_h / vis_shape[1]))
                    elif click_point == None:
                        click_point = (-1, -1)    # Which indicates no click point.
                    frame_keypoint_list.append(click_point)
                frame_keypoint = np.array(frame_keypoint_list)
                frame_keypoint_name = frame_key + '_keypoint'
                if frame_keypoint_name in h5py_file.keys(): del h5py_file[frame_keypoint_name]
                h5py_file[frame_keypoint_name] = frame_keypoint
    
if __name__ == '__main__':
    droid_root = '/home/cvte/twilight/home/data/droid_h5py'
    droid_grasp_frame(droid_root, start_idx = 50055)
    #annotate_graspframe_gripper(droid_root, start_idx = 0)