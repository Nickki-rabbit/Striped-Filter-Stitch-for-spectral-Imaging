from __future__ import division
from PIL import Image
from matplotlib import pyplot as plt
import cv2
import numpy as np
import time
import os
import pdb
import scipy.io as sio

# parameters
start_time = time.time()
path = '/Users/liuy/Desktop/0830_2/'
files = os.listdir(path[0:-1])
datacube_name = 'lab2.mat'
datacube_key = datacube_name[0:-4]
cliped_y0 = 400
cliped_y1 = 1840
height = 2048
cut_w = 30
# coor number
dim = 30

files.sort()
num_img = 363
# imread all images into 3 parts
image = [cv2.imread(path + img_file, 0) for img_file in files]

runpoint = time.time()

templt_1 = cv2.imread('./template/template_1_1.png', 0)
templt_2 = cv2.imread('./template/template_2.png', 0)
templt_3 = cv2.imread('./template/template_3.png', 0)

num_img_1 = 117
num_img_2 = 115
#num_img_3 = 202
#num_img = num_img_1 + num_img_2 + num_img_3

# cut coordinates: 30
coor = [
    141, 228, 305, 387, 475, 557, 641, 720, 794, 869, 955, 1031, 1116, 1187,
    1264, 1340, 1435, 1499, 1570, 1647, 1733, 1812, 1885, 1961, 2042, 2130,
    2202, 2278, 2354, 2436
]
# the coordinates of the upleft corner of the image
upleft = []
# the shape of final stitch result pano
# pano = [[[] for i in range(4850)] for i in range(2048)]
# store the cut part
cut = []
# the distance(diff) between cut[i] and cut[0]
diff = []


def get_match_locX(img, templt, last_locX):
    if last_locX is None:
        match = cv2.matchTemplate(img, templt, cv2.TM_CCOEFF_NORMED)
        _, _, _, maxLoc = cv2.minMaxLoc(match)
        maxLocX = maxLoc[0]
    else:
        h, w = templt.shape
        move_range = 30
        img = img[:, last_locX - move_range:last_locX + w]
        match = cv2.matchTemplate(img, templt, cv2.TM_CCOEFF_NORMED)
        _, _, _, tmpLoc = cv2.minMaxLoc(match)
        maxLocX = last_locX - move_range + tmpLoc[0]
    return maxLocX


# calculate diff
last_locX_1, last_locX_2, last_locX_3 = None, None, None
for i in range(num_img):
    print 'image:' + str(i + 1) + ':'
    t0 = time.time()
    # part 1: match template 1
    if i < num_img_1:
        maxLocX = get_match_locX(image[i], templt_1, last_locX_1)
        last_locX_1 = maxLocX
        upleft.append(maxLocX)
        diff.append(np.absolute(upleft[0] - upleft[i]))
        print '(' + str(diff[i]) + ')'
    # part 2: match template_2
    elif i < num_img_1 + num_img_2:
        if i == num_img_1:
            maxLocX_temp = get_match_locX(image[i], templt_1, last_locX_1)
            temp = np.absolute(maxLocX_temp - upleft[0])
        maxLocX = get_match_locX(image[i], templt_2, last_locX_2)
        last_locX_2 = maxLocX
        upleft.append(maxLocX)
        diff.append(temp + np.absolute(upleft[num_img_1] - upleft[i]))
        print '(' + str(diff[i]) + ')'
    # part 3: match template 3
    else:
        if i == num_img_1 + num_img_2:
            maxLocX_temp = get_match_locX(image[i], templt_2, last_locX_2)
            temp_2 = np.absolute(maxLocX_temp - upleft[num_img_1])
        maxLocX = get_match_locX(image[i], templt_3, last_locX_3)
        last_locX_3 = maxLocX
        upleft.append(maxLocX)
        diff.append(temp_2 + temp +
                    np.absolute(upleft[num_img_1 + num_img_2] - upleft[i]))
        print '(' + str(diff[i]) + ')'
    print('elisp time: %f s' % (time.time() - t0))

new_images = [image[0][:, cor:cor + cut_w] for cor in coor]
new_image_stack = np.stack(new_images, axis=-1)
cover_times = np.ones(new_image_stack.shape[1])
for i in range(1, num_img):
    overlap_size = cut_w - (diff[i] - diff[i - 1])
    cut_stack = np.stack(
        [image[i][:, cor:cor + cut_w] for cor in coor], axis=-1)
    overlap_part1 = new_image_stack[:, diff[i]:, :]
    overlap_part2 = cut_stack[:, :overlap_size, :]
    # update cover times
    cover_times[diff[i]:] += 1
    cover_times = np.hstack([cover_times, np.ones(cut_w - overlap_size)])
    # average
    new_image_stack[:, diff[i]:, :] = overlap_part1 * \
        (cover_times[diff[i]:] - 1) * 1. / cover_times[diff[i]:] + \
        overlap_part2 * 1. / cover_times[diff[i]:]
    new_image_stack = np.concatenate(
        (new_image_stack, cut_stack[:, overlap_size:, :]), axis=1)

# clip to the same field
remain_size = new_image_stack.shape[1] - (coor[-1] - coor[0])
cliped_images = []
for i in range(dim):
    start_point = coor[-1] - coor[i]
    end_point = start_point + remain_size
    new_image = new_image_stack[:, :, i]
    cliped_images.append(new_image[cliped_y0:cliped_y1, start_point:end_point])
cliped_images = np.stack(cliped_images, axis=-1)

# save mat & png
sio.savemat(datacube_name,
            {datacube_key: cliped_images.astype('uint8')})
img = Image.fromarray(cliped_images[:, :, 0].astype('uint8'))
img.save(datacube_key + '.png')
print('total time: %f' % (time.time() - start_time))
