import keras.models
from keras.applications import VGG16
from keras.preprocessing import image
import numpy as np
from ReStart.Codes.setKerasSession import setKerasAllow_Groth_lof_device_placement

setKerasAllow_Groth_lof_device_placement()

# vgg16 = VGG16(include_top=False, weights='imagenet')
#
# model_path = r'C:\Users\NagyMiklosZoltan\PycharmProjects\Szakdolgozat2020\ReStart\Models\Full_5_ClassModel.h5'
# my_model = keras.models.load_model(model_path)
#
# path = 'C:\\Users\\NagyMiklosZoltan\\PycharmProjects\\Szakdolgozat2020\\algonautsChallenge2019\\Training_Data\\'
#
# classes: list = []
# for i in range(1, 10):
#     img_path = path + '92_Image_Set\\92images\\image_0' + str(i) + '.jpg'
#     # predicting images
#     img = image.load_img(img_path, target_size=(175, 175))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#
#     images = np.vstack([x])
#     output = vgg16.predict(images)
#     classes.append(my_model.predict_classes(output))
#
# print(classes)

model_path = r'C:\Users\NagyMiklosZoltan\PycharmProjects\Szakdolgozat2020\ReStart\Models\Full_5_ClassModel.h5'
my_model = keras.models.load_model(model_path)