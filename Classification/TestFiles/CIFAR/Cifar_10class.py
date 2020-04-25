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
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense, Input, LeakyReLU
from keras.callbacks import ModelCheckpoint
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

train_data_dir = \
    r'C:\Users\NagyMiklosZoltan\PycharmProjects\Szakdolgozat2020\RawImages\CIFAR_Images\train'
validation_data_dir = \
    r'C:\Users\NagyMiklosZoltan\PycharmProjects\Szakdolgozat2020\RawImages\CIFAR_Images\validation'

start = datetime.datetime.now()

batch_size = 128
num_class = 10

datagen = ImageDataGenerator(rescale=1. / 255,
                             rotation_range=20,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             horizontal_flip=True)

train_gen = datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

valid_gen = datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

m_Input = Input(shape=(img_width, img_height, 3))
vgg16 = applications.VGG16(include_top=False,
                           input_tensor=m_Input,
                           weights='imagenet')
# print(vgg16.summary())

for layer in vgg16.layers[:3]:
    layer.trainable = False

print(vgg16.layers[-1].output)
x = Flatten()(vgg16.layers[-1].output)
x = Dense(100)(x)
x = LeakyReLU(alpha=0.3)(x)
x = Dropout(0.3)(x)
x = Dense(50)(x)
x = LeakyReLU(alpha=0.3)(x)
x = Dropout(0.3)(x)
last_layer = Dense(num_class, activation='softmax')(x)

model = Model(inputs=vgg16.input, outputs=last_layer)

print(model.summary())

learning_rate = 0.1
lr_decay = 1e-6
lr_drop = 20

def lr_scheduler(epoch):
    return learning_rate * (0.5 ** (epoch // lr_drop))


reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)
sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['acc'])

# checkpoint
filepath = "weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min', period=1)
callbacks_list = [checkpoint]

num_train = len(train_gen.filenames)
num_valid = len(valid_gen.filenames)

history = model.fit_generator(
    train_gen,
    steps_per_epoch=num_train // batch_size,
    epochs=10,
    validation_data=valid_gen,
    validation_steps=num_valid // batch_size,
    callbacks=[reduce_lr]
)

print(history.history.keys())
model.save_weights('model_Weights.h5')
model.save("full_model_cifar_10class.hdf5")

end = datetime.datetime.now()
elapsed = end - start
print('Time: ', elapsed)
plot_history(history)
