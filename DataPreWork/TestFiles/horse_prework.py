import cv2
from os import listdir
from os.path import isfile, join

input_image = 'C:\\Users\\NagyMiklosZoltan\\PycharmProjects\\Szakdolgozat2020\\RawImages\\horse-or-human\\validation\\horses'
output_path = input_image

imageSource = [join(input_image, f) for f in listdir(input_image) if isfile(join(input_image, f))]

for item in imageSource:
    img = cv2.imread(item)
    img = cv2.resize(src=img, dsize=(175, 175), interpolation=cv2.INTER_AREA)
    cv2.imwrite(item, img)