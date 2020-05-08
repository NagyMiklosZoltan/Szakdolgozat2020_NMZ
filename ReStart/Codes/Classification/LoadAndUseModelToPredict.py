import keras.models
from keras.preprocessing import image
import numpy as np
import cv2


def GetPrediction(image_path, model_path):
    my_model = keras.models.load_model(model_path)
    image_list = image.list_pictures(image_path)
    # print(image_list[0])
    # print(len(image_list))

    images: list = []
    for im in image_list:
        img = cv2.imread(im)
        img = cv2.resize(img, (175, 175))
        images.append(img)

    data = np.array(images)

    a = my_model.predict(data, batch_size=16)
    print('Predikció kész!')
    a = np.argmax(a, axis=1)

    return a
