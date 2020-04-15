import cv2
from os import listdir
from os.path import isfile, join
from DataPreWork import PreWork


def preworkAllDirs():
    input_image = 'C:\\Users\\NagyMiklosZoltan\\PycharmProjects\\Szakdolgozat2020\\RawImages\\horse-or-human\\train\\'
    output_path = input_image

    # Gauss Blur kernel size
    dim = 3
    k_size = (dim, dim)
    # Edge Detecition Thresholds
    thresholds = 100, 100
    # Expected Size to resize
    size = (64, 64)

    preWork = PreWork.PreWork(g_kernel=k_size, ex_size=size, thresholds=thresholds)

    preWork.all_resize_local(input_path=input_image)

    input_image = 'C:\\Users\\NagyMiklosZoltan\\PycharmProjects\\Szakdolgozat2020\\RawImages\\horse-or-human\\validation\\'
    preWork.all_resize_local(input_path=input_image)

preworkAllDirs()