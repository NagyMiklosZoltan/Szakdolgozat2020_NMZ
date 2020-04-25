import random
import os
import shutil
import math
import cv2

from ReStart.Codes.directories import dataset_dir_dict, train_dir_dict, valid_dir_dict


# Returns unique random elements from a list
# random.sample()

def getImagesWithPath(r):
    return [os.path.join(r, f) for f in os.listdir(r)]


def getMinSample():
    min_s = 100000
    for k in dataset_dir_dict.keys():
        k_dir = dataset_dir_dict[k]
        count = len([name for name in os.listdir(k_dir)])
        print(count)
        min_s = min(min_s, count)

    # round down min_sample with two decimal places
    return math.ceil(min_s // 100) * 100


def getRandomFiles(files, k):
    return random.sample(files, k)


def resizeImages(files, size):
    for file in files:
        img = cv2.imread(file)
        img = cv2.resize(src=img, dsize=size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(file, img)


def copyResizedFilesTrainValid(src, dest_key, size):
    train_dest = train_dir_dict[dest_key]
    valid_dest = valid_dir_dict[dest_key]

    # split validation data 10%
    scale = len(src) // 10
    # VALIDATION
    print('Copy Files to Validation')
    for file in src[:scale]:
        shutil.copy(file, valid_dest)

    new_files = getImagesWithPath(valid_dest)
    resizeImages(new_files, size)

    # TRAIN
    print('Copy Files to Train')
    for file in src[scale:]:
        shutil.copy(file, train_dest)

    print('Start Resizing')
    new_files = getImagesWithPath(train_dest)
    resizeImages(new_files, size)


# **********************************************************************************************************
# PROGRAM
num_sample = getMinSample()

for key in dataset_dir_dict.keys():
    print('Current Class:' + key)
    key_dir = dataset_dir_dict[key]
    curr_images = getImagesWithPath(r=key_dir)
    print('Got images:' + str(len(curr_images)))

    sample = getRandomFiles(files=curr_images, k=num_sample)
    print('Samples.....DONE')

    size = 175, 175
    copyResizedFilesTrainValid(src=sample, dest_key=key, size=size)
    print('***' + key + '***')
