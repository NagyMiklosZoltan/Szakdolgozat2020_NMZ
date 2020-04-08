import PIL as pl
import cv2

# # With PIL
# img = pl.Image.open('C:\\Users\\NagyMiklosZoltan\\PycharmProjects\\Szakdolgozat2020\\DataPreWork\\TestFiles\\image_010.jpg')
# size = 175, 175
# img.thumbnail(size)
# img.save("C:\\Users\\NagyMiklosZoltan\\PycharmProjects\\Szakdolgozat2020\\DataPreWork\\TestFiles\\image_010_resized.jpg", "JPEG")

# With OpenCV
img = cv2.imread('C:\\Users\\NagyMiklosZoltan\\PycharmProjects\\Szakdolgozat2020\\DataPreWork\\TestFiles\\image_010.jpg')
# print(img.shape)
size = 175, 175
img = cv2.resize(src=img, dsize=size, interpolation=cv2.INTER_AREA)
cv2.imwrite("C:\\Users\\NagyMiklosZoltan\\PycharmProjects\\Szakdolgozat2020\\DataPreWork\\TestFiles\\image_010_resized_cv2.jpg", img)