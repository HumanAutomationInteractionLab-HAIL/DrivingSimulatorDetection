import cv2


face_cascade = cv2.CascadeClassifier('D:\github\DrivingSimulatorDetection\CarDetection\cascade.xml')
vc = cv2.VideoCapture('D:\github\DrivingSimulatorDetection\CarDetection\car_test.mp4')

if vc.isOpened():
    rval , frame = vc.read()
else:
    rval = False

while rval:
    rval, frame = vc.read()
    # car detection.
    cars = face_cascade.detectMultiScale(frame,1.2,5,cv2.CASCADE_SCALE_IMAGE,)
    
    ncars = 0
    for (x,y,w,h) in cars:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        ncars = ncars + 1

    # show result
    cv2.imshow("Result",frame)
    cv2.waitKey(0);
    #vc.release()
