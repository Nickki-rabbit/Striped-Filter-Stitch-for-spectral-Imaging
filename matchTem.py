import cv2
import numpy as np
import glob
import time

start = time.time()
num_img = 12
image = [cv2.imread('./stitches/pano_' + str(i+1) + '.png', 0) for i in range(num_img) ]
templt = cv2.imread('./template/datacube.png', 0)


w, h = templt.shape

for i in range(num_img):
    mtch = cv2.matchTemplate(image[i], templt, cv2.TM_CCOEFF_NORMED) # mtch is a single-channel image, which is easier to analysis
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(mtch)
    #print maxLoc
    top_left = maxLoc
    bottom_right = (top_left[0] + h, top_left[1] + w)

    cv2.rectangle(image[i], top_left, bottom_right, 255, 2)
    cv2.imwrite('images_' + str(i+1) + '.png', image[i])
    cv2.imwrite('datacube_' + str(i+1) + '.png', image[i][top_left[1]: top_left[1]+w, top_left[0]:top_left[0] + h])

elapse = time.time() - start
print('datacube run time: ',str(elapse))
