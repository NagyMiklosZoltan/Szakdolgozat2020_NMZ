import cv2
from os import path, listdir
from os.path import isfile, join

class PreWork:

    def __init__(self, g_kernel, ex_size, thresholds):
        self.g_kernel = g_kernel
        self.ex_size = ex_size
        self.thresholds = thresholds

    def ImagePreProcessing(self, input_path, output_path):
        imageSource = [f for f in listdir(input_path) if isfile(join(input_path, f))]
        thre_1, thre_2 = self.thresholds
        for item in imageSource:
            # Read in GrayScale
            img = cv2.imread(input_path + '\\' + item, 0)
            # Resize Image if not expected
            if (img.shape[0], img.shape[1]) != self.ex_size:
                img = cv2.resize(src=img, dsize=self.ex_size, interpolation=cv2.INTER_AREA)
            # Gauss Blurr
            gaus = cv2.GaussianBlur(img, self.g_kernel, cv2.BORDER_DEFAULT)
            # Edge Extract
            gaus_edged = cv2.Canny(gaus, thre_1, thre_2)
            cv2.imwrite(output_path + '\\' + item, gaus_edged)