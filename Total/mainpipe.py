from laneDetectionMethods import *

#white_output = 'Sample.mp4'
#clip1 = VideoFileClip("test_videos/Sample2.mp4")
#white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
#white_clip.write_videofile(white_output, audio=False)

def triarea(p1,p2,p3):
    (x1,y1)=p1
    (x2,y2)=p2
    (x3,y3)=p3
    return x1y2-x1y3+x2y3-x2y1+x3y1-x2y2

def inarea(point,leftpoints,rightpoints):
    x,y=point
    (lbx,lby),(lux,luy)=leftpoints
    (rbx,rby),(rux,ruy)=rightpoints
    s1=triarea(point,(lbx,lby),(lux,luy))
    s2=triarea(point,(lux,luy),(rux,ruy))
    s3=triarea(point,(rux,ruy),(rbx,rby))
    s4=triarea(point,(rbx,rby),(lbx,lby))
    totals=triarea((lux,luy),(lbx,lby),(rux,ruy))+triarea((rux,ruy),(rbx,rby),(lbx,lby))
    if (s1+s2+s3+s4-totals)>0.001:
        return False
    else:
        return True


cap=cv2.VideoCapture("Sample.mp4")
timeF=3#3帧抽取一次
c=1
totalFrameNumber = cap.get(cv2.CAP_PROP_FRAME_COUNT)
cutframe=int(totalFrameNumber/timeF)
lp=[];
rp=[];
ffourcc = cv2.VideoWriter_fourcc(*'XVID')
out=cv2.VideoWriter('output_opends.avi',ffourcc,15,(1920,1080))
while(cap.isOpened()):
    ret,frame=cap.read()
    if(ret==True):
        if c%timeF==0:
            frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            result,leftpoints,rightpoints=process_image(frame)
            lp.append(leftpoints)#[(左下x,左下y)，(左上x,左上y)]
            rp.append(rightpoints)#[(右下x,右下y)，(右上x,右上y)]
            result=cv2.cvtColor(result,cv2.COLOR_BGR2RGB)
            out.write(result)
            print(c,'/',totalFrameNumber)
    else:
        break
    c=c+1
cap.release()
out.release()
cv2.destroyAllWindows()


