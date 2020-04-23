import cv2
import scipy.io as sio
import numpy as np
from os import mkdir
from os.path import isdir

matfile = 'C:\\Users\\NagyMiklosZoltan\\Downloads\\stl10_matlab.tar\\stl10_matlab\\test.mat'
images = sio.loadmat(matfile)

print(images.keys())

# for train.mat
# # print(np.shape(images['class_names'])) (1, 10) 0-tól indexelve
# # print(np.shape(images['X'])) (5000, 27648)
# # print(np.shape(images['y'])) (5000, 1) Egy és 10 közötti értékek
# # print(images['class_names'][0][0][0])
# # print(images['class_names'][0][1][0])
# # print(images['class_names'][0][2][0])
# # print(images['class_names'][0][3][0])
# # print(images['y'][3][0])
# # print(np.unique(images['y'][:])) [ 1  2  3  4  5  6  7  8  9 10]

path = 'C:\\Users\\NagyMiklosZoltan\\PycharmProjects\\Szakdolgozat2020\\RawImages\\stl10_imageSet\\validation'
# print(np.unique(images['y'][:]))


for i in range(0, 8000):
    arr_1d = np.array(images['X'][i]).flatten()
    arr_3d = arr_1d.reshape((3, 96, 96)).transpose()
    class_idx = images['y'][i][0]
    # print(class_idx)
    dirname = images['class_names'][0][class_idx-1][0]
    # print(dirname)
    output_path = path + '\\' + dirname
    if not isdir(output_path):
        mkdir(output_path)
    cv2.imwrite(output_path + '\\matv_%d.jpg' % i, arr_3d)
    print(i)



# 1 ['airplane']
# 2 ['bird']
# 3 ['car']
# 4 ['cat']
# 5 ['deer']
# 6 ['dog']
# 7 ['horse']
# 8 ['monkey']
# 9 ['ship']
# 10 ['truck']

# dict_keys(['__header__', '__version__', '__globals__', 'X', 'y', 'class_names', 'fold_indices'])
# 500 training
# 800 test