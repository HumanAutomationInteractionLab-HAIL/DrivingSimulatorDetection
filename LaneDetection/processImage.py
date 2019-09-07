from laneDetectionMethods import *
import cv2
#Test on a single image
#test_images = [read_image('test_images/' + i) for i in os.listdir('test_images/')]
test_images=[cv2.cvtColor(cv2.imread('test_opends.png'),cv2.COLOR_BGR2RGB)]
draw_lane_lines(test_images[0])
plt.show()