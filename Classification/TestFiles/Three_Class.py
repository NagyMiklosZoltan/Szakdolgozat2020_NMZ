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

from Classification.TestFiles.PlotResults import plot_history
from Classification.TestFiles.setKerasSession import setKerasAllow_Groth_lof_device_placement

setKerasAllow_Groth_lof_device_placement()

# Default dimensions we found online
img_width, img_height = 64, 64

# Create a bottleneck file
top_model_weights_path = 'bottleneck_fc_model.h5'  # loading up our datasets

train_data_dir = \
    'C:\\Users\\NagyMiklosZoltan\\PycharmProjects\\Szakdolgozat2020\\RawImages\\horse-or-human\\train'
validation_data_dir = \
    'C:\\Users\\NagyMiklosZoltan\\PycharmProjects\\Szakdolgozat2020\\RawImages\\horse-or-human\\validation'

# number of epochs to train top model
epochs = 7  # this has been changed after multiple model run
# batch size used by flow_from_directory and predict_generator


# Loading vgc16 model
vgg16 = applications.VGG16(include_top=False, weights='imagenet')
datagen = ImageDataGenerator(rescale=1. / 255)
# needed to create the bottleneck .npy files

# __this can take an hour and half to run so only run it once.
# once the npy files have been created, no need to run again. Convert this cell to a code cell to run.__

start = datetime.datetime.now()

batch_size = 128

generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)

nb_train_samples = len(generator.filenames)
# num_classes = len(generator.class_indices)

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

# ********************************************************************************************************************

# training data
generator_top = datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

nb_train_samples = len(generator_top.filenames)
num_classes = len(generator_top.class_indices)

# load the bottleneck features saved earlier
train_data = np.load('bottleneck_features_train.npy')

# get the class labels for the training data, in the original order
train_labels = generator_top.classes

# convert the training labels to categorical vectors
train_labels = to_categorical(train_labels, num_classes=num_classes)

# Validation data
generator_top_valid = datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

nb_valid_samples = len(generator_top_valid.filenames)
num_classes = len(generator_top_valid.class_indices)

# load the bottleneck features saved earlier
validation_data = np.load('bottleneck_features_valid.npy')

# get the class labels for the training data, in the original order
validation_labels = generator_top_valid.classes

# convert the training labels to categorical vectors
validation_labels = to_categorical(validation_labels, num_classes=num_classes)

# This is the best model we found. For additional models, check out I_notebook.

start = datetime.datetime.now()

model = Sequential()
model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dense(100, activation=keras.layers.LeakyReLU(alpha=0.3)))
model.add(Dropout(0.4))
model.add(Dense(50, activation=keras.layers.LeakyReLU(alpha=0.3)))
model.add(Dropout(0.4))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc', 'loss'])

history = model.fit(train_data, train_labels,
                    epochs=120,
                    batch_size=128,
                    validation_data=(validation_data, validation_labels))
print(history.history.keys())
model.save_weights(top_model_weights_path)

(eval_loss, eval_accuracy) = model.evaluate(
    validation_data, validation_labels,
    batch_size=batch_size,
    verbose=1)

print('[INFO] accuracy: {:.2f}%'.format(eval_accuracy * 100))
print('[INFO] Loss: {}'.format(eval_loss))
end = datetime.datetime.now()
elapsed = end - start
print('Time: ', elapsed)
plot_history(history)

