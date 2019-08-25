import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
from collections import deque

def shibietest(img1,kp1,des1,img2):#识别提取视频帧中是否存在指定图像
    MIN_MATCH_COUNT = 20
    time_start=time.time()
# Initiate SIFT detector

    sift = cv2.xfeatures2d.SIFT_create()
#sift=cv2.SIFT()
# find the keypoints and descriptors with SIFT
    kp2, des2 = sift.detectAndCompute(img2,None)


    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
    good = []

    good_min=0
    
    for m,n in matches:
        if m.distance < 0.8*n.distance:
            good.append(m) 

    if len(good)==0:
        return [0,0]

    locator_x=[]
    locator_y=[]
    for mat in good:
  #  img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx
        # x - columns
        # y - rows
   # (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt
        locator_x.append(x2)
        locator_y.append(y2)
        ##print(x2,y2)

    
    
    n=0
    count=len(locator_x)
  
    locator=[]
    while(n<count):
        (x,y)=(int(locator_x[n]),int(locator_y[n]))
        locator.append((x,y))
        n=n+1
    count=len(locator)
    
    
    locator_new = []
    for id in locator:
        if id not in locator_new:
            locator_new.append(id)

    count=len(locator_new)
    
    #locator_new=locator
    
    locator_x=[]
    locator_y=[]
    
    x1=450# 后视镜坐标
    y1=500# 后视镜坐标
    x2=900
    y2=480
    n=0
    while(n<count):
        (x,y)=locator_new[n]
        dis1=np.abs(x1-x)+np.abs(y1-y)
        dis2=np.abs(x2-x)+np.abs(y2-y)
       # print(locator_x[n],locator_y[n]
        
        if(dis1<100 or dis2<100):
            del locator_new[n]            
            count=count-1
            n=n-1
        else:
            locator_x.append(x)
            locator_y.append(y)
            
        n=n+1
    n=0

    
    count=len(locator_new)
   
    
    
    #滤波
    B=1
    
    locator_mean=(np.mean(locator_x),np.mean(locator_y))
    locator_std=(np.std(locator_x),np.std(locator_y))
    while(n<count):
        min_x=locator_mean[0]-B*locator_std[0]
        max_x=locator_mean[0]+B*locator_std[0]
        min_y=locator_mean[1]-B*locator_std[1]
        max_y=locator_mean[1]+B*locator_std[1]
        
        if((locator_x[n]<min_x) or (locator_x[n]>max_x) or (locator_y[n]<min_y) or (locator_y[n]>max_y)):
            del locator_x[n]
            del locator_y[n]
            n=n-1
            count=count-1
        n=n+1
        
    count=len(locator_x)
    print(count)
    print(locator_x)
    print(locator_y)
    if(count<8):
        return [0,0]
    
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)




    #draw_params = dict(matchColor = (0,255,0), # draw matches in green color
    #               singlePointColor = None,
    #               matchesMask = matchesMask, # draw only inliers
    #               flags = 2)

    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None)#,**draw_params)
    plt.imshow(img3, 'gray'),plt.show()
    
    time_end=time.time()
    print(time_end-time_start)

    

    locator_mean=[np.mean(locator_x),np.mean(locator_y)]
    
    
    print(locator_mean)
    
    
    return (locator_mean)
def track(old_img,new_img,point):#光流法追踪图像在相邻帧中的位置
    
    [x,y]=point
    x=float(x)
    y=float(y)

    area=2
    point=np.array([[[x,y]],[[x+area,y+area]],[[x+area,y-area]],[[x-area,y+area]],[[x-area,y-area]]],dtype='float32')
    #cv2.imshow('frame',new_img)
    
    lk_params = dict( winSize  = (20,20),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # 创建随机颜色
    color = np.random.randint(0,255,(100,3))

    old_gray=cv2.cvtColor(old_img,cv2.COLOR_BGR2GRAY)
    frame_gray=cv2.cvtColor(new_img,cv2.COLOR_BGR2GRAY)
    pl,st,err=cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, point, None, **lk_params)

    
    good_new = pl[st==1]
    good_old = point[0]
    
    mask = np.zeros_like(old_img)

    [x,y]=[0,0]
    for [x0,y0] in good_new:
        x=x+x0
        y=y+y0
    
    [x,y]=[x/len(good_new),y/len(good_new)]
    

    good_new=np.array([[x,y]],dtype='float32')


    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv2.circle(new_img,(a,b),5,color[i].tolist(),-1)
    img = cv2.add(frame,mask)

    cv2.imshow('frame',img)
    
    cv2.waitKey(0)
    
    
    
   

    return [x,y]

cap2=cv2.VideoCapture("F:\\Recodring\Sample.mp4")#原数据
img1 = cv2.imread('right_signword.jpg',0)          # queryImage
sift = cv2.xfeatures2d.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1,None)

timeF=6#隔6帧取一次
count_d=10
c=1


lower_orange=np.array([240,120,60])
upper_orange=np.array([255,160,100])
result1=[]
result2=[]

buff_detect=deque()
buff_size=10
buff_count=0

while(cap2.isOpened() and (c<3600)):
  ret2, frame3 = cap2.read()
 
  
  if (ret2 == True) and (c%timeF==0):

    if buff_count>count_d:   #save for detect
      buff_count=buff_count-1
      buff_detect.popleft()
      
    buff_detect.append(frame3)
    buff_count=buff_count+1
    
  #if ret==True:
    frame2=cv2.cvtColor(frame3,cv2.COLOR_BGR2RGB)
    cv2.imshow('Origin',frame3)
   # result1.append(center(mask))
    point_result2=shibietest(img1,kp1,des1,frame2)
    result2.append(point_result2)

    if point_result2 != [0,0]:                     # detect back
        result2_size=len(result2)
        buff_detect.pop()
        buff_detect_size=len(buff_detect)
        old_img=frame2

        buff_count=0
        print(buff_detect_size)
        for i in range(buff_detect_size):
            new_img=buff_detect.pop()
            point=result2[result2_size-1-i]
            print(point)
            
            point_result2=track(old_img,new_img,point)
            
            print(point_result2)
            
            result2[result2_size-2-i]=point_result2
            old_img=new_img.copy()

    
    order=cv2.waitKey(10)
    if order == ord('s'):
      cv2.waitKey(0)
    elif order== ord('q'):
      break

  c=c+1


cap2.release()

cv2.destroyAllWindows()
"""
img1 = cv2.imread('left_g.png',0)
img2 = cv2.imread('zuosign.png',0)
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1,None)
shibietest(img1,kp1,des1,img2)
"""



