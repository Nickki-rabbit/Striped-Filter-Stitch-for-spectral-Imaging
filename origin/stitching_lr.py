from __future__ import division
from PIL import Image
from scipy.misc import imsave
from matplotlib import pyplot as plt
import cv2
import numpy as np
import time
import os
import pdb
import scipy.io as sio
from matplotlib.backends.backend_pdf import PdfPages

# parameters
start_time = time.time()
path = 'first_time/'
template_path = 'template_2/'
files = os.listdir(path[0:-1])
files.sort()
files = files[4:]
datacube_name = 'lab2.mat'
datacube_key = datacube_name[0:-4]
cliped_y0 = 400
cliped_y1 = 1840
cut_w = 30
dim = 30  # coor number

template_loc = [[(226, 97), (414, 1354)], [(110, 77), (1205, 1380)],
                [(1101, 116), (1270, 1684)]]
template_from = [
    files[4], 'WCF_20170823145856005800206.bmp',
    'WCF_20170823145856005800413.bmp'
]
# cut coordinates: 30
coor = [
    141, 228, 305, 387, 475, 557, 641, 720, 794, 869, 955, 1031, 1116, 1187,
    1264, 1340, 1435, 1499, 1570, 1647, 1733, 1812, 1885, 1961, 2042, 2130,
    2202, 2278, 2354, 2436
]

# imread all images into 3 parts
# average image
img0 = cv2.imread(path + files[0], 0)
avg_img = img0.astype(np.float64)
for img_file in files[1:]:
    avg_img = avg_img + cv2.imread(path + img_file, 0)
avg_img = avg_img / len(files)
height, width = avg_img.shape

runpoint = time.time()
pdf = PdfPages('check_2.pdf')


def get_template(img, topleft, lowright, save_name):
    x0, y0 = topleft
    x1, y1 = lowright
    template = img[y0:y1, x0:x1]
    imsave(save_name, template)
    return template


def get_match_loc(img, templt, last_loc, img_id):
    img0 = img
    h, w = templt.shape
    if last_loc is None:
        match = cv2.matchTemplate(img, templt, cv2.TM_CCOEFF_NORMED)
        _, maxV, _, maxLoc = cv2.minMaxLoc(match)
    else:
        move_range = 28
        y0 = max(0, last_loc[1] - move_range)
        y1 = min(img.shape[0], last_loc[1] + h + move_range)
        x0 = last_loc[0]
        x1 = last_loc[0] + w + move_range
        img = img[y0:y1, x0:x1]
        match = cv2.matchTemplate(img, templt, cv2.TM_CCOEFF_NORMED)
        _, maxV, _, tmpLoc = cv2.minMaxLoc(match)
        maxLocX = x0 + tmpLoc[0]
        maxLocY = y0 + tmpLoc[1]
        maxLoc = (maxLocX, maxLocY)
    print(maxV)
    if maxV < 0.80:
        fig, axis = plt.subplots()
        axis.imshow(img0, cmap='gray')
        rect = plt.Rectangle(maxLoc, w, h, edgecolor='r', facecolor='none')
        axis.add_patch(rect)
        axis.set_title('image: %s' % img_id)
        pdf.savefig()
        fig, axis = plt.subplots()
        axis.imshow(img, cmap='gray')
        pdf.savefig()
    return maxLoc


def drop_img_avg(img, avg_img):
    img = img - avg_img
    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype('uint8')
    return img


# get template
templts = []
for i in range(len(template_loc)):
    img = cv2.imread(path + template_from[i], 0) - avg_img
    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype('uint8')
    loc = template_loc[i]
    templts.append(
        get_template(img, loc[0], loc[1], template_path +
                     'template_%d.png' % i))

last_locs = [None, None, None]
upleft_last = 0
templt_id = 0
upleft_referer = None
referer_move = 0
# ######## initialize
img = cv2.imread(path + files[0], 0)
img_avg = drop_img_avg(img, avg_img)
maxLoc = get_match_loc(img_avg, templts[templt_id], last_locs[templt_id], 0)
last_locs[templt_id] = maxLoc
upleft_referer = maxLoc[0]
upleft_last = maxLoc[0]
# the distance(diff) between cut[i] and cut[0]
diff = [0]
new_images = [img[:, cor:cor + cut_w] for cor in coor]
new_image_stack = np.stack(new_images, axis=-1)
cover_times = np.ones(new_image_stack.shape[1])

for i in range(1, len(files)):
    t0 = time.time()
    print('image: %d' % i)
    img = cv2.imread(path + files[i], 0)
    img_avg = drop_img_avg(img, avg_img)
    # check whether the template going out of camera
    templt_w = template_loc[templt_id][1][0] - template_loc[templt_id][0][0]
    if width - (upleft_last + templt_w) < 100:
        # change template
        maxLoc_temp = get_match_loc(img_avg, templts[templt_id],
                                    last_locs[templt_id], i)
        referer_move = upleft_referer - maxLoc_temp[0]
        templt_id += 1
        # stop the loop
        if templt_id + 1 > len(templts):
            break
    maxLoc = get_match_loc(img_avg, templts[templt_id], last_locs[templt_id],
                           i)
    last_locs[templt_id] = maxLoc

    if width - (upleft_last + templt_w) < 100:
        # change upleft referer
        upleft_referer = maxLoc[0] + referer_move

    # NOTE: diff is minus
    diff.append(upleft_referer - maxLoc[0])
    upleft_last = maxLoc[0]
    print '(' + str(diff[i]) + ')'
    print('elasp time: %f s' % (time.time() - t0))

    # ############### cut & average ############
    overlap_size = cut_w - (diff[i - 1] - diff[i])
    if overlap_size == cut_w:
        print("image doesn't move")
        break
    assert overlap_size < cut_w
    cut_stack = np.stack([img[:, cor:cor + cut_w] for cor in coor], axis=-1)
    overlap_part1 = new_image_stack[:, diff[i] - overlap_size:diff[i], :]
    overlap_part2 = cut_stack[:, -overlap_size:, :]
    # update cover times
    cover_times[diff[i] - overlap_size:diff[i]] += 1
    cover_times = np.hstack([np.ones(cut_w - overlap_size), cover_times])
    # average
    overlap_coeff = cover_times[diff[i] - overlap_size:diff[i]]
    overlap_coeff = overlap_coeff.reshape([-1, 1])
    new_image_stack[:, diff[i]-overlap_size:diff[i], :] = overlap_part1 * \
        (overlap_coeff - 1) * 1. / overlap_coeff + \
        overlap_part2 * 1. / overlap_coeff
    new_image_stack = np.concatenate(
        (cut_stack[:, :-overlap_size, :], new_image_stack), axis=1)

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
for i in range(dim):
    fig, axis = plt.subplots()
    axis.imshow(cliped_images[:, :, i], cmap='gray')
    pdf.savefig()
pdf.close()
