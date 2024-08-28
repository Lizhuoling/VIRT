import cv2
import h5py
import pdb
import os
import numpy as np

import tkinter as tk
from tkinter import filedialog, Canvas
import cv2
import numpy as np
from PIL import Image, ImageTk

class DataSegmentAnnotationTool:
    def __init__(self, root, data_root,):
        self.root = root
        self.data_root = data_root

        self.h5py_path = os.path.join(self.data_root, 'h5py')
        self.cam_names = ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
        
        self.root.title("Video Segment Annotation Tool")
        self.root.geometry("2048x876")
        self.root.option_add('*Dialog.msg.font', 'Helvetica 14')

        # Set when specifying the video.
        self.h5py_filepath = None
        self.cap = None
        self.total_frames = 0
        # Set when specifying a annotation instance.
        self.current_frame = 0   # The current frame Number
        # Save the result.
        self.segment_labels = None
        
        # The canvas to show images
        self.canvas = tk.Canvas(root)
        self.canvas.pack(padx=20, pady=20)
        self.canvas.bind("<Button-1>", self.on_click)

        self.control_panel = tk.Frame(root)
        self.control_panel.pack()

        self.load_button = tk.Button(self.control_panel, text="Load", command=self.load)
        self.load_button.pack(side="left")
        self.load_button.config(font=("Helvetica", 16), height=2, width=12)
        self.load_button.pack(pady=10)

        self.prev_button = tk.Button(self.control_panel, text="<< Prev", command=self.prev_frame)
        self.prev_button.pack(side="left")
        self.prev_button.config(font=("Helvetica", 16), height=2, width=12)
        self.prev_button.pack(pady=10)

        self.next_button = tk.Button(self.control_panel, text="Next >>", command=self.next_frame)
        self.next_button.pack(side="left")
        self.next_button.config(font=("Helvetica", 16), height=2, width=12)
        self.next_button.pack(pady=10)

        self.annotate_button = tk.Button(self.control_panel, text="Annotate", command=self.annotate)
        self.annotate_button.pack(side="left")
        self.annotate_button.config(font=("Helvetica", 16), height=2, width=12)
        self.annotate_button.pack(pady=10)

        self.clean_button = tk.Button(self.control_panel, text="Clean", command=self.clean)
        self.clean_button.pack(side="left")
        self.clean_button.config(font=("Helvetica", 16), height=2, width=12)
        self.clean_button.pack(pady=10)

        self.frame_slider = tk.Scale(self.control_panel, orient="horizontal", command=self.slide_frame, length=600,)
        self.frame_slider.pack(fill="x")

        seg_label = tk.Label(root, text="Segment ID", font = ("Helvetica", 16))
        seg_label.pack(padx=10, pady=5)
        self.seg_label_entry = tk.Entry(root)
        self.seg_label_entry.pack(padx=10, pady=5)
        self.seg_label_entry.insert(0, "-1")

        self.cam_selection_var = tk.StringVar()
        for cam_name in self.cam_names:
            btn = tk.Radiobutton(root, text=cam_name, variable=self.cam_selection_var, value=cam_name, font = ("Helvetica", 16))
            btn.pack(side=tk.LEFT, anchor='w', padx=10)

    def load(self):
        self.segment_labels = None
        self.h5py_filepath = filedialog.askopenfilename(initialdir = self.h5py_path)
        if self.h5py_filepath:
            cam_selection = self.cam_selection_var.get()
            if cam_selection == '': cam_selection = 'cam_left_wrist'

            h5py_filename = self.h5py_filepath.rsplit('/')[-1]
            video_filename = h5py_filename.replace('.hdf5', '.mp4')
            video_path = os.path.join(self.data_root, cam_selection, video_filename)

            self.cap = cv2.VideoCapture(video_path)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_slider.config(to=self.total_frames - 1)

            h5py_f = h5py.File(self.h5py_filepath, 'r')
            if 'seg_keyframe' in h5py_f.keys():
                self.segment_labels = h5py_f['seg_keyframe'][:]
            else:
                self.segment_labels = None
            h5py_f.close()

            self.show_frame(self.current_frame)

    def show_frame(self, frame_no):
        frame_no = int(frame_no)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame_no
            
            if self.segment_labels is not None:
                if self.current_frame < self.segment_labels[0][0]: 
                    status = 0
                elif self.current_frame >= self.segment_labels[0][0] and self.current_frame < self.segment_labels[-1][0]:
                    for i in range(self.segment_labels.shape[0] - 1):
                        if self.current_frame >= self.segment_labels[i][0] and self.current_frame < self.segment_labels[i + 1][0]:
                            status = self.segment_labels[i][1]
                            break
                else:
                    status = self.segment_labels[-1][1]

                font_scale = 1
                thickness = 2
                text_size = cv2.getTextSize(str(status), cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                text_x = 10
                text_y = text_size[1] + 10
                cv2.putText(frame, str(status), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness, lineType=cv2.LINE_AA)

            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.photo = ImageTk.PhotoImage(image=image)
            self.canvas.config(width=self.photo.width(), height=self.photo.height())
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    def on_click(self, event):
        pass

    def prev_frame(self):
        if self.h5py_filepath == None:
            print('Please load a h5py file first.')
            return
        if self.current_frame > 0:
            self.current_frame -= 1
            self.show_frame(self.current_frame)
            self.frame_slider.set(self.current_frame)

    def next_frame(self):
        if self.h5py_filepath == None:
            print('Please load a h5py file first.')
            return
        if self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self.show_frame(self.current_frame)
            self.frame_slider.set(self.current_frame)

    def slide_frame(self, value):
        if self.h5py_filepath == None:
            print('Please load a h5py file first.')
            return
        self.current_frame = int(value)
        self.show_frame(self.current_frame)

    def annotate(self,):
        seg_label = int(self.seg_label_entry.get())
        if seg_label == -1:
            print('Please first input a segment label.')
            return
        new_annotation = np.array([[self.current_frame, seg_label],], dtype = np.int32)

        if self.segment_labels is None:
            self.segment_labels = new_annotation
        else:
            self.segment_labels = np.concatenate((self.segment_labels, new_annotation), axis=0)

        h5py_f = h5py.File(self.h5py_filepath, 'r+')
        if 'seg_keyframe' in h5py_f.keys(): del h5py_f['seg_keyframe']
        h5py_f['seg_keyframe'] = self.segment_labels
        h5py_f.close()
        print("Annotated.")

        self.show_frame(self.current_frame)

    def clean(self):
        if self.segment_labels is not None:
            if self.current_frame < self.segment_labels[0][0]: 
                print("The initial stage does not need to clean.")
                return
            else:
                if self.current_frame >= self.segment_labels[0][0] and self.current_frame < self.segment_labels[-1][0]:
                    for i in range(self.segment_labels.shape[0] - 1):
                        if self.current_frame >= self.segment_labels[i][0] and self.current_frame < self.segment_labels[i + 1][0]:
                            row_to_delete = i
                            break
                else:
                    row_to_delete = self.segment_labels.shape[0] - 1
                self.segment_labels = np.delete(self.segment_labels, row_to_delete, axis=0)
                if self.segment_labels.shape[0] == 0:
                    self.segment_labels = None
                h5py_f = h5py.File(self.h5py_filepath, 'r+')
                if 'seg_keyframe' in h5py_f.keys(): del h5py_f['seg_keyframe']
                if self.segment_labels != None: h5py_f['seg_keyframe'] = self.segment_labels
                h5py_f.close()

        else:
            print("No annotations to clean.")
            return
        
        self.show_frame(self.current_frame)


if __name__ == '__main__':
    root_path = '/home/cvte/twilight/data/aloha_beverage'
    root = tk.Tk()
    app = DataSegmentAnnotationTool(root, root_path)
    root.mainloop()
    