from laneDetectionMethods import *

#white_output = 'Sample.mp4'
#clip1 = VideoFileClip("test_videos/Sample2.mp4")
#white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
#white_clip.write_videofile(white_output, audio=False)

cap=cv2.VideoCapture("sample_opends.mp4")
timeF=3#隔3帧截取一次
c=1
totalFrameNumber = cap.get(cv2.CAP_PROP_FRAME_COUNT)
ffourcc = cv2.VideoWriter_fourcc(*'XVID')
out=cv2.VideoWriter('output_opends.avi',ffourcc,15,(1920,1080))
while(cap.isOpened()):
    ret,frame=cap.read()
    if(ret==True):
        if c%timeF==0:
            frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            result=process_image(frame)
            result=cv2.cvtColor(result,cv2.COLOR_BGR2RGB)
            out.write(result)
            print(c,'/',totalFrameNumber)
    else:
        break
    c=c+1
cap.release()
out.release()
cv2.destroyAllWindows()

