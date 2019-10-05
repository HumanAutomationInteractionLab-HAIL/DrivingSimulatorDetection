def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:  # if it is not gray-scale
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


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

    #cv2.imshow('frame',img)
    
    #cv2.waitKey(0)
    return (x,y)
