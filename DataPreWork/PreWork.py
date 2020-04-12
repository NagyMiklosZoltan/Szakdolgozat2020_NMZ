import cv2
from os import path, listdir, mkdir
from os.path import isfile, isdir, join

class PreWork:

    def __init__(self, g_kernel, ex_size, thresholds):
        self.g_kernel = g_kernel
        self.ex_size = ex_size
        self.thre_1, self.thre_2 = thresholds

    def getSubDirectories(self, input_path):
        s_dirs = [dir for dir in listdir(input_path) if isdir(join(input_path, dir))]
        return s_dirs

    def preProcess(self, input_image):
        img = cv2.imread(input_image, 0)
        # Resize Image if not expected
        if (img.shape[0], img.shape[1]) != self.ex_size:
            img = cv2.resize(src=img, dsize=self.ex_size, interpolation=cv2.INTER_AREA)
        # Gauss Blurr
        gaus = cv2.GaussianBlur(img, self.g_kernel, cv2.BORDER_DEFAULT)
        # Edge Extract
        gaus_edged = cv2.Canny(gaus, self.thre_1, self.thre_2)

        return gaus_edged

    def folder_image_pre_processing(self, input_path, output_path):
        imageSource = [f for f in listdir(input_path) if isfile(join(input_path, f))]
        for item in imageSource:
            img = self.preProcess(input_image=input_path + '\\' + item)
            if not isdir(output_path):
                mkdir(output_path)
            cv2.imwrite(output_path + '\\' + item, img)

    def image_pre_processing(self, input_path, output_path):
        s_dirs = self.getSubDirectories(input_path=input_path)
        for dir in s_dirs:
            in_dir = input_path + '\\' + dir
            out_dir = output_path + '\\' + dir
            self.folder_image_pre_processing(input_path=in_dir, output_path=out_dir)
        print('PreProcess Complete for all given subfolders (' + ', '.join(s_dirs) + ')')