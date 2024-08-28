from ultralytics import YOLOv10

import pdb
import cv2

yolov10 = YOLOv10('/home/agilex/twilight/code/act_project/act/pretrained/yolov10_aloha_beverage.pt')

cap = cv2.VideoCapture(0)

cnt = 0
while cap.isOpened():
    cnt += 1

    ret, frame = cap.read()

    result = yolov10.predict(source=frame, imgsz=640, conf=0.25)[0]
    frame = result.plot()

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()