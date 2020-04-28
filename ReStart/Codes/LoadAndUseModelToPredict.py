import keras.models
from keras.applications import VGG16
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from ReStart.Codes.setKerasSession import setKerasAllow_Groth_lof_device_placement
import os
import cv2

setKerasAllow_Groth_lof_device_placement()

vgg16 = VGG16(include_top=False, weights='imagenet')

model_path = r'C:\Users\NagyMiklosZoltan\PycharmProjects\Szakdolgozat2020\ReStart\Models\Full_5_ClassModel.h5'
my_model = keras.models.load_model(model_path)

path = 'C:\\Users\\NagyMiklosZoltan\\PycharmProjects\\Szakdolgozat2020\\algonautsChallenge2019\\Training_Data\\'

classes: list = []
datagen = ImageDataGenerator(rescale=1. / 255)

img_path = r'C:\Users\NagyMiklosZoltan\PycharmProjects\Szakdolgozat2020\algonautsChallenge2019\Training_Data\92_Image_Set\92images'

image_list = image.list_pictures(img_path)
print(len(image_list))

images: list = []
for im in image_list:
    img = cv2.imread(im)
    img = np.reshape(img, [175, 175, 3])
    images.append(img)

data = np.array(images)

print(np.shape(data))


files = datagen.flow(data, batch_size=16, shuffle=False)

output = vgg16.predict_generator(files, steps=92)

np.save('for_prediction.npy', output)
print('Save done')
predict_data = np.load('for_prediction.npy')
print(np.shape(predict_data))
a = my_model.predict_classes(predict_data, batch_size=1)

for i in range(92):
    print(a[i])
