import cv2
from os import path, listdir
from os.path import isfile, join

inputDirectory = "92images"

allImageSource = [f for f in listdir(inputDirectory) if isfile(join(inputDirectory, f))]
print(allImageSource)

# tried gauss kernel size: (21,21)

dim = 21
k_size = (dim, dim)


for item in allImageSource:
    src = cv2.imread(r'92images\\' + item, cv2.IMREAD_UNCHANGED)
    dst = cv2.GaussianBlur(src, k_size, cv2.BORDER_DEFAULT)
    cv2.imwrite(r'G_92images\G_' + item, dst)