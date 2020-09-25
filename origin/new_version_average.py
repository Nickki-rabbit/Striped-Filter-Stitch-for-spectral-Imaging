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
path = './images/'
files = os.listdir(path[0:-1])
datacube_name = 'qd_sample.mat'
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
# average image
img0 = cv2.imread(path + files[0], 0)
avg_img = img0.astype(np.float64)
for img_file in files[1:]:
    avg_img = avg_img + cv2.imread(path + img_file, 0)
avg_img = avg_img / len(files)
image = [cv2.imread(path + img_file, 0) for img_file in files]

runpoint = time.time()

templt_1 = cv2.imread('./template/template_1_1.png', 0)
templt_2 = cv2.imread('./template/template_2.png', 0)
templt_3 = cv2.imread('./template/template_3.png', 0)
templts = [templt_1, templt_2, templt_3]
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
last_locXs = [None, None, None]
upleft_last = 0
templt_id = 0
upleft_referer = None
referer_move = 0
for i in range(num_img):
    img_i = cv2.imread(path + files[i], 0) - avg_img
    img_i = ((img_i - img_i.min()) / (img_i.max() - img_i.min()) * 255).astype('uint8')
    print 'image:' + str(i + 1) + ':'
    t0 = time.time()

    # change template
    if i > 0 and upleft_last < 100:
        maxLocX_temp = get_match_locX(img_i, templts[templt_id],
                                      last_locXs[templt_id])
        referer_move = upleft_referer - maxLocX_temp
        templt_id += 1

    maxLocX = get_match_locX(img_i, templts[templt_id],
                             last_locXs[templt_id])

    last_locXs[templt_id] = maxLocX
    # change upleft referer
    if upleft_last < 100:
        upleft_referer = maxLocX + referer_move
    diff.append(upleft_referer - maxLocX)
    upleft_last = maxLocX
    print(upleft_last)
    print '(' + str(diff[i]) + ')'
    print('elisp time: %f s' % (time.time() - t0))

img0 = cv2.imread(path + files[0], 0)
new_images = [img0[:, cor:cor + cut_w] for cor in coor]
new_image_stack = np.stack(new_images, axis=-1)
cover_times = np.ones(new_image_stack.shape[1])
for i in range(1, num_img):
    img_i = cv2.imread(path + files[i], 0)
    overlap_size = cut_w - (diff[i] - diff[i - 1])
    cut_stack = np.stack(
        [img_i[:, cor:cor + cut_w] for cor in coor], axis=-1)
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
sio.savemat(datacube_name, {datacube_key: cliped_images.astype('uint8')})
img = Image.fromarray(cliped_images[:, :, 0].astype('uint8'))
img.save(datacube_key + '.png')
print('total time: %f' % (time.time() - start_time))
