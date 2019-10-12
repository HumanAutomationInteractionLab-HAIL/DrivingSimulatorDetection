from CarDetectionMethod_s import *
import matplotlib.pyplot as plt
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
from collections import deque
import copy
import re

def takeFirst(elem):
    return elem[0]

def writetotxt(data,filename):
    b=[str(i) for i in data]
    with open(filename,'w') as f:
        for line in b:
            f.write(line+'\n')
    return 

def readfromtxt(filename):
    b=[]
    with open(filename,'r') as f:
        lines=f.readlines()
        for i in lines:
            for j in ',()':
                i=i.replace(j,' ')
            templine=i.split()
            index,x1,y1,x2,y2,xc,yc=[float(j) for j in templine]
            b.append((index,(x1,y1),(x2,y2),(xc,yc)))
    return b


def unique(frameout):
    frameout.sort(key=takeFirst)
    templist1=[]
    templist2=[]
    i=0
    j=0
    while i<len(frameout)-2:
        tempindex=frameout[i][0]
        for j in range(i+1,len(frameout)):
            if frameout[j][0]!=tempindex:
                if j-i<2:
                    break
                else:                    
                    templist1=copy.deepcopy(frameout[i:j])
                    for k in range(i,j):
                        frameout.pop(i)
                    templist2=centerunique(templist1)
                    for k in templist2:
                        frameout.insert(i,k)
                    i=i+len(templist2)-1
                    break
        i=i+1
    return frameout

def distance(p1,p2):
    (x1,y1)=p1
    (x2,y2)=p2
    return ((x1-x2)**2+(y1-y2)**2)**0.5

def centerunique(frameout):
    templist=[]
    (index_g,(x1_g,y1_g),(x2_g,y2_g),(xg,yg))=frameout[0]
    for i,(index,(x1,y1),(x2,y2),(xc,yc)) in enumerate(frameout):
        if distance((xc,yc),(xg,yg))>50:
            templist.append(frameout[i])
        else:
            x1_g=(x1+x1_g*i)/(i+1)
            x2_g=(x2+x2_g*i)/(i+1)
            y1_g=(y1+y1_g*i)/(i+1)
            y2_g=(y2+y2_g*i)/(i+1)
            xg=(xc+xg*i)/(i+1)
            yg=(yc+yg*i)/(i+1)

    templist.append((index_g,(x1_g,y1_g),(x2_g,y2_g),(xg,yg)))
    i=0
    while i<len(templist)-2:
        (tempxc1,tempyc1)=templist[i][3]
        for j in range(i+1,len(templist)):
            (tempxc2,tempyc2)=templist[j][3]
            if distance((tempxc1,tempyc1),(tempxc2,tempyc2))<50:
                templist.insert(0,templist[i])
                templist.insert(0,templist[j])
                return centerunique(templist)
        i=i+1
    return templist
        


                   

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
            if (w>0.4*h) and (w<1.6*h) and w>20 and h>20 and w<400 and h<250:
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


for i in range(timeF-1):
    frameout.pop(0)
c=0
frameout.sort(key=takeFirst)
writetotxt(frameout,'original.txt')
frameout=unique(frameout)
writetotxt(frameout,'unique.txt')
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


    