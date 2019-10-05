from CarDetectionMethod_s import *
import matplotlib.pyplot as plt
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
from collections import deque
import copy


def takeFirst(elem):
    return elem[0]

face_cascade = cv2.CascadeClassifier('D:\github\DrivingSimulatorDetection\CarDetection\cascade.xml')
vc = cv2.VideoCapture('D:\github\DrivingSimulatorDetection\CarDetection\car_test.mp4')
cap = cv2.VideoCapture('D:\github\DrivingSimulatorDetection\CarDetection\car_test.mp4')
road_v = np.array([[(0, 1050), (620, 500), (1200, 500), (1900, 1050)]], dtype=np.int32)#过滤路mask
if vc.isOpened():
    rval , frame = vc.read()
else:
    rval = False
timeF=3 #隔5帧取一次
count_d=20#回溯帧数
buff_detectleft=deque(maxlen=count_d)
buff_detectright=deque(maxlen=count_d)
frameout=[(0,(0,0),(0,0)),(0,0)]
c=0
while rval:
    c=c+1
    rval, frame = vc.read()
    post_frame = vc.get(1)      
    buff_detectleft.append(frame)
    if c%timeF==0:
        # car detection.
        newframe=region_of_interest(frame,road_v)
        cars = face_cascade.detectMultiScale(newframe,1.2,6,cv2.CASCADE_SCALE_IMAGE,)
    
        ncars = 0
        for (x,y,w,h) in cars:
            if (h>0.3*w) and (h<2*w) and w>20 and h>20:
                #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
                for i in range(count_d):
                    cap.set(cv2.CAP_PROP_POS_FRAMES,int(post_frame+i))
                    a,b=cap.read()
                    if a:
                        buff_detectright.append(b)
                frameout.append((post_frame,(x,y),(x+w,y+h),(x+w/2,y+h/2)))           
                ncars = ncars + 1
                for i in range(count_d-1):
                    [x1,y1]=track(buff_detectleft[count_d-i-1],buff_detectleft[count_d-i-2],[x,y])
                    [x2,y2]=track(buff_detectleft[count_d-i-1],buff_detectleft[count_d-i-2],[x+w,y+h])
                    if(x1==-1) or (x2==-1):
                        continue
                    else:
                        frameout.append((post_frame-i-1,(x1,y1),(x2,y2),((x1+x2)/2,(y1+y2)/2)))

                for i in range(len(buff_detectright)-1):
                    [x1,y1]=track(buff_detectright[i],buff_detectright[i+1],[x,y])
                    [x2,y2]=track(buff_detectright[i],buff_detectright[i+1],[x+w,y+h])
                    if(x1==-1) or (x2==-1):
                        continue
                    else:
                        frameout.append((post_frame+i+1,(x1,y1),(x2,y2),((x1+x2)/2,(y1+y2)/2)))
            
# show result
vc.release()
vc = cv2.VideoCapture('D:\github\DrivingSimulatorDetection\CarDetection\car_test.mp4')
ffourcc = cv2.VideoWriter_fourcc(*'XVID')
out=cv2.VideoWriter('output.avi',ffourcc,30,(1920,1080))
if vc.isOpened():
    rval , frame = vc.read()
else:
    rval = False
frameout.sort(key=takeFirst)
for i in range(timeF-1):
    frameout.pop(0)
c=0
(index,(x1,y1),(x2,y2),(xc,yc))=frameout[0]
while rval:
    c=c+1
    rval, frame = vc.read()
    post_frame = vc.get(1)
    while index==post_frame:
        cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),2)
        frameout.pop(0)
        if (len(frameout)):
           (index,(x1,y1),(x2,y2),(xc,yc))=frameout[0]
        else:
            break
    out.write(frame)
    print(c)


    