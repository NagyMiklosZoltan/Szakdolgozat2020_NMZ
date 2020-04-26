import pandas as pd
import numpy as np
import itertools
import keras
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras import optimizers
from keras.preprocessing import image
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import datetime
import time

from ReStart.Codes.PlotResults import plot_history
from ReStart.Codes.setKerasSession import setKerasAllow_Groth_lof_device_placement
from ReStart.Codes.directories import dataset_dir_dict, train_dir_dict, valid_dir_dict

def preTraining():
    setKerasAllow_Groth_lof_device_placement()

    # Default dimensions we found online
    img_width, img_height = 175, 175

    modelpath =  r'C:\Users\NagyMiklosZoltan\PycharmProjects\Szakdolgozat2020\ReStart\Models'
    # Create a bottleneck file
    top_model_weights_path = modelpath + '\\' + 'bottleneck_fc_model.h5'  # loading up our datasets

    train_data_dir = train_dir_dict['base_dir']
    validation_data_dir = valid_dir_dict['base_dir']

    # number of epochs to train top model
    epochs = 7  # this has been changed after multiple model run

    vgg16 = applications.VGG16(include_top=False, weights='imagenet')
    datagen = ImageDataGenerator(rescale=1. / 255)

    start = datetime.datetime.now()

    batch_size = 16

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    nb_train_samples = len(generator.filenames)
    num_classes = len(generator.class_indices)

    predict_size_train = int(math.ceil(nb_train_samples / batch_size))
    print(nb_train_samples / batch_size)
    bottleneck_features_train = vgg16.predict_generator(generator, predict_size_train)

    np.save('bottleneck_features_train.npy', bottleneck_features_train)
    end = datetime.datetime.now()
    elapsed = end - start
    print('Time: ', elapsed)


    generator_valid = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    nb_validation_samples = len(generator_valid.filenames)
    # num_classes = len(generator.class_indices)

    predict_size_validation = int(math.ceil(nb_validation_samples / batch_size))

    bottleneck_features_validation = vgg16.predict_generator(generator_valid, predict_size_validation)

    np.save('bottleneck_features_valid.npy', bottleneck_features_validation)
    end = datetime.datetime.now()
    elapsed = end - start
    print('Time: ', elapsed)


preTraining()