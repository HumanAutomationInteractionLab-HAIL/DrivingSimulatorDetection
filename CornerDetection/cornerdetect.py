import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
from collections import deque

def cornerdetect(dst):
    shape = dst.shape
    height = shape[0]
    width = shape[1]
    corner=[]

    for i in range(height):
        for j in range(width):
           if dst[i][j][0]== 225:
               corner.append((i,j))
    return corner

#def fourcorners(corner):
#    (x,y)=corner[0]
#    leftup=x+y
#    rightup=x-y
#    leftdown=x-y
#    rightdown=x+y
#    leftupP=0
#    rightupP=0
#    leftdownP=0
#    rightdownP=0
#    for k in range(len(corner)):
#        (tempx,tempy)=corner[k]
#        sum1=tempx+tempy
#        sum2=tempx-tempy
#        if sum1>leftup:
#            leftup=sum1
#            leftupP=k
#        if sum1<rightdown:
#            rightdown=sum1
#            rightdownP=k
#        if sum2>rightup:
#            rightup=sum2
#            rightupP=k
#        if sum2<leftdown:
#            leftdown=sum2
#            leftdownP=k
#    finalcorner=[corner[leftupP],corner[rightupP],corner[leftdownP],corner[rightdownP]]
#    return finalcorner

def fourcorners(corner):
    finalcorner=[]
    (y,x)=corner[0][0]
    leftup=x+y
    rightup=x-y
    leftdown=x-y
    rightdown=x+y
    leftupP=0
    rightupP=0
    leftdownP=0
    rightdownP=0
    for k in range(len(corner)):
        a=corner[k][0]
        sum1=a[1]+a[0]
        sum2=a[1]-a[0]
        if sum1>leftup:
            leftup=sum1
            leftupP=k
        if sum1<rightdown:
            rightdown=sum1
            rightdownP=k
        if sum2>rightup:
            rightup=sum2
            rightupP=k
        if sum2<leftdown:
            leftdown=sum2
            leftdownP=k
    (x,y)=corner[leftupP][0]
    finalcorner.append((y,x))
    (x,y)=corner[rightupP][0]
    finalcorner.append((y,x))
    (x,y)=corner[leftdownP][0]
    finalcorner.append((y,x))    
    (x,y)=corner[rightdownP][0]
    finalcorner.append((y,x))
    return finalcorner

def showpoint(img,point):
    (x,y)=point
    x=int(x)
    y=int(y)
    img[x][y][0]=0
    img[x][y][1]=255
    img[x][y][2]=0
    return img

def exchange(point):
    (x,y)=point
    return (y,x)






h=800#图像宽度
w=1100#图像高度
cap1=cv2.VideoCapture("F:\\OpencvTest\OpenCV1.avi")#眼动仪数据
timeF=6#隔6帧取一次
count_d=30
c=1

##滤色接口
#lower_orange=np.array([240,120,60])
#upper_orange=np.array([255,160,100])
#result1=[]
#result2=[]

while(cap1.isOpened() and (c<3600)):
  ret1, frame1 = cap1.read()
  if (ret1 == True) and (c%timeF==0):
   #if ret==True:
    frame2=frame1[0:h,0:w]
    gray=cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    avgthresh=np.mean(blurred)
    #print(avgthresh)
    #print(gray[1][1])
    #print(gray[480][640])
    thresh = cv2.threshold(blurred,avgthresh+10,255,cv2.THRESH_BINARY)[1]
    
    cv2.imshow('thresh',thresh)
    #cv2.imshow('Gray',gray)
    #cv2.imshow('Origin',frame1)
    #cv2.imshow('blurred',blurred)
    dst=np.float32(thresh)

    #Haris Detect
    #dst = cv2.cornerHarris(thresh, 2, 3, 0.06)
   
    #cv2.imshow('dst', dst)
    #img=np.copy(frame2)                    
    #img[dst > 0.0001*dst.max()] = [225, 0, 0]
    
    #cv2.imshow('Harris', img)

    #cornerpoint=cornerdetect(img)

    #GoodFeatureDetect
    corner=cv2.goodFeaturesToTrack(thresh,0,0.001,1)
    cornerpoint=fourcorners(corner)
    frame3=np.copy(frame2)
  
    for point in cornerpoint:
        frame3=showpoint(frame3,point)
    cv2.line(frame3,exchange(cornerpoint[0]),exchange(cornerpoint[1]),(255, 0, 0), 1)
    cv2.line(frame3,exchange(cornerpoint[1]),exchange(cornerpoint[3]),(255, 0, 0), 1)
    cv2.line(frame3,exchange(cornerpoint[3]),exchange(cornerpoint[2]),(255, 0, 0), 1)
    cv2.line(frame3,exchange(cornerpoint[2]),exchange(cornerpoint[0]),(255, 0, 0), 1)
    cv2.imshow('point',frame3)

    order=cv2.waitKey(10)
    if order == ord('s'):
      cv2.waitKey(0)
    elif order== ord('q'):
      break

  c=c+1


cap1.release()

cv2.destroyAllWindows()




