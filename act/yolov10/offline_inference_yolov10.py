from ultralytics import YOLOv10

import pdb
import cv2

yolov10 = YOLOv10('pretrained/yolov10_aloha_beverage.pt')

cap = cv2.VideoCapture("/home/agilex/twilight/data/aloha_beverage/aloha_beverage_multicam_detvideo/cam_left_wrist/episode_0.mp4")
if not cap.isOpened():   
    print("Failed to open camera")
    exit()

width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
video_writter = cv2.VideoWriter("demo.mp4", fourcc, 20, (width, height))

cnt = 0
while cap.isOpened():
    cnt += 1
    if cnt >= cap.get(cv2.CAP_PROP_FRAME_COUNT): break
    
    ret, frame = cap.read()

    result = yolov10.predict(source=frame, imgsz=640, conf=0.25)[0]
    frame = result.plot()
    video_writter.write(frame)
    #cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
video_writter.release()