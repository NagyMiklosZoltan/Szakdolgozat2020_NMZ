import cv2
from os import path, listdir
from os.path import isfile, join

# InputImages
inputDirectory = r'C:\Users\NagyMiklosZoltan\PycharmProjects\Szakdolgozat2020\92images'
allImageSource = [f for f in listdir(inputDirectory) if isfile(join(inputDirectory, f))]

# Gauss Blur kernel size
dim = 3
k_size = (dim, dim)

# Edge Detecition Thresholds
thre_1 = 100
thre_2 = 150

# Expected Size to resize
size = (175, 175)

outputDirectory = r'C:\Users\NagyMiklosZoltan\PycharmProjects\Szakdolgozat2020\G_92images'

# Read in GrayScale
for item in allImageSource:
    img = cv2.imread(inputDirectory + '\\' + item, 0)
    # Resize Image if not expected
    if size != (img.shape[0], img.shape[1]):
        img = cv2.resize(src=img, dsize=size, interpolation=cv2.INTER_AREA)

    gaus = cv2.GaussianBlur(img, k_size, cv2.BORDER_DEFAULT)
    gaus_edged = cv2.Canny(gaus, thre_1, thre_2)
    cv2.imwrite(outputDirectory + '\\G_' + item, gaus_edged)