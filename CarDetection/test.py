from CarDetectionMethod_s import *
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
from collections import deque

face_cascade = cv2.CascadeClassifier('D:\github\DrivingSimulatorDetection\CarDetection\cascade.xml')
vc = cv2.VideoCapture('D:\github\DrivingSimulatorDetection\CarDetection\car_test.mp4')
road_v = np.array([[(150, 1050), (820, 640), (1020, 640), (1780, 1050)]], dtype=np.int32)#过滤路mask
if vc.isOpened():
    rval , frame = vc.read()
else:
    rval = False
timeF=5 #隔5帧取一次
count_d=10#回溯帧数
buff_detect=deque(maxlen=count_d)
frameout=[(0,(0,0),(0,0))]
c=0
while rval:
    c=c+1
    rval, frame = vc.read()
    post_frame = vc.get(1)      
    buff_detect.append(frame)
    if c%timeF==0:
        # car detection.
        newframe=region_of_interest(frame,road_v)
        cars = face_cascade.detectMultiScale(newframe,1.2,5,cv2.CASCADE_SCALE_IMAGE,)
    
        ncars = 0
        for (x,y,w,h) in cars:
            if (h>0.3*w) and (h<2*w) and w>20 and h>20:
                #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)              
                ncars = ncars + 1
                for i in range(count_d-1):
                    [x1,y1]=track(buff_detect[count_d-i],buff_detect[count_d-i-1],[x,y])
                    [x2,y2]=track(buff_detect[count_d-i],buff_detect[count_d-i-1],[x+w,y+h])
                    frameout.append((post_frame-i-1,(x1,y1),(x2,y2))

    # show result
    vc.release()

