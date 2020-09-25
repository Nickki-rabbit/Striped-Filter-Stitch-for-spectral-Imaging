from __future__ import division
from PIL import Image
from matplotlib import pyplot as plt
import cv2
import numpy as np
import time
import os
import pdb
import scipy.io as sio

start = time.time()
num_img = 41
# imread all images into 3 parts
# image = [cv2.imread('/Users/nickki/Documents/Spec905/image_stitching/images/push_2/5_0830_2/WCF_20170830102406015400' + str(i+468) + '.bmp', 0) for i in range(num_img) ]

# image = [cv2.imread('../20190319_10mhz_1.8vpp/WCF_201903191737160991000' + str(i+71) + '.bmp', 0) for i in range(num_img) if i < 29]
# for i in range(7):
#    image.append(cv2.imread('../20190319_10mhz_1.8vpp/WCF_20190319173716099100' + str(i+100) + '.bmp', 0))
# image = [cv2.imread('../20190322_100fps_0.01hz_0.42vpp/WCF_20190322214221057300' + str(i+862) + '.bmp', 0) for i in range(num_img)]

# image = [cv2.imread('../20190319_10mhz_1.8vpp/WCF_20190319173716099100' + str(i+71) + '.bmp', 0)]


# path = '/Users/nickki/Documents/Spec905/system/test/201903/20190322_100fps_0.01hz_0.42vpp/'
# path = '/Users/nickki/Documents/Spec905/system/test/201903/20190327_100fps_0.02hz_0.47vpp/'
# path = '/Users/nickki/Documents/Spec905/system/test/202001/wcf/'
path = './wcf/'

files = os.listdir(path[0:-1])
files.sort()
image = [cv2.imread(path + img_file, 0) for img_file in files]

runpoint = time.time()

templt_1 = cv2.imread('./template/temp_3.bmp', 0)
templt_2 = cv2.imread('./template/temp_1.tif', 0)
# templt_2 = cv2.imread('../template/template_8.png', 0)
# templt_3 = cv2.imread('../template/template_8.png', 0)

num_img_1 = 20
num_img_2 = 21
#num_img_3 = 202
#num_img = num_img_1 + num_img_2 + num_img_3

# cut QD stripe width
width = 30
# cut coordinates: 6
coor = [
        89, 112, 137, 158, 179, 200, 220, 241, 261, 281, 303, 321
        ]

# the coordinates of the upleft corner of the image
upleft = []
# the shape of final stitch result pano
pano =[[[] for i in range(2048)] for i in range(1024)]
# store the cut part
cut = []
# the distance(diff) between cut[i] and cut[0]
diff = []

for cor in coor:
    # only the first QD stripe go through the feature match
    if coor.index(cor) == 0:
        for i in range(num_img):
            print('image:', str(i+1) ,':')
            # part 1: match template 1
            if i < num_img_1:
                # print 'i = :' + str(i+1)
                mtch = cv2.matchTemplate(image[i+1], templt_1, cv2.TM_CCOEFF_NORMED)
                minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(mtch)
                upleft.append(maxLoc[0])
                diff.append(np.absolute(upleft[0] - upleft[i]))
                print ('(', str(diff[i]), ')')
            # part 2: match template_2
            elif i < num_img_1 + num_img_2:
                if i == num_img_1:
                    match = cv2.matchTemplate(image[i], templt_1, cv2.TM_CCOEFF_NORMED)
                    minVal, maxVal, minLoc, maxLoc_temp = cv2.minMaxLoc(match)
                    temp = np.absolute(maxLoc_temp[0] - upleft[0])
                mtch = cv2.matchTemplate(image[i], templt_2, cv2.TM_CCOEFF_NORMED)
                minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(mtch)
                upleft.append(maxLoc[0])
                diff.append(temp +  np.absolute(upleft[num_img_1] - upleft[i]))
                print('(', str(diff[i]), ')')
            # part 3: match template 3
            else:
                if i == num_img_1 + num_img_2:
                    match = cv2.matchTemplate(image[i], templt_2, cv2.TM_CCOEFF_NORMED)
                    minVal, maxVal, minLoc, maxLoc_temp = cv2.minMaxLoc(match)
                    temp_2 = np.absolute(maxLoc_temp[0] - upleft[num_img_1])
                mtch = cv2.matchTemplate(image[i], templt_3, cv2.TM_CCOEFF_NORMED)
                minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(mtch)
                upleft.append(maxLoc[0])
                diff.append(temp_2 + temp + np.absolute(upleft[num_img_1 + num_img_2] - upleft[i]))
                print( '(', str(diff[i]), ')')
            # pile up to pano
            cut = image[i+1][:, cor:cor+width]
            for j in range(1024):
                for k in range(width):
                    pano[j][k + diff[i]].append(cut[j][k])
        # the final stitching result: do average
        diff_diff = [diff[i] - diff[i-1] for i in range(len(diff))]
        print('max_diff_diff is: ', str(max(diff_diff)))
        pano_result = np.zeros((1024, diff[-1]+width))
        for i in range(1024):
                for j in range(diff[-1] + width):
                    # print 'the length of pano is: ' + str(len(pano[i][j]))
                    if len(pano[i][j]) == 0:
                        pano_result[i, j] = sum(pano[i][j])
                    else:
                        pano_result[i, j] = sum(pano[i][j]) / len(pano[i][j])
        # convert and save the result
        formatted = (pano_result * 255 / np.max(pano_result)).astype('uint8')
        img = Image.fromarray(formatted)
        img.save('./stitches/pano_' + str(coor.index(cor)+1) + '.png')
        print( ' #########Already stitch 1 #############')
        pano =[[[] for i in range(2048)] for i in range(1024)]
        # the 2nd to the 30th QD stripe
    else:
        for i in range(num_img):
            cut = image[i+1][:, cor:cor+width]
            for j in range(1024):
                for k in range(width):
                    pano[j][k + diff[i]].append(cut[j][k])
        pano_result = np.zeros((1024, diff[-1]+width))
        for i in range(1024):
            for j in range(diff[-1] + width):
                if len(pano[i][j]) == 0:
                    pano_result[i, j] = sum(pano[i][j])
                else:
                    pano_result[i, j] = sum(pano[i][j]) / len(pano[i][j])
        formatted = (pano_result * 255 / np.max(pano_result)).astype('uint8')
        img = Image.fromarray(formatted)
        img.save('./stitches/pano_' + str(coor.index(cor)+1) + '.png')
        print(' #########Already stitch ', str(coor.index(cor)+1), ' #############')
        pano =[[[] for i in range(2048)] for i in range(1024)]


all_elapse = (time.time() - start)
run_elapse = (time.time() - runpoint)
print('all time: ', str(all_elapse), ', run time: ', str(run_elapse))
