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
from keras.layers import Dropout, Flatten, Dense, Input, LeakyReLU, Conv2D, BatchNormalization, MaxPool2D
from keras.activations import relu
from keras.optimizers import rmsprop
from keras.callbacks import ModelCheckpoint

from Classification.TestFiles.PlotResults import plot_history
from Classification.TestFiles.setKerasSession import setKerasAllow_Groth_lof_device_placement

setKerasAllow_Groth_lof_device_placement()

train_data_dir = \
    r'C:\Users\NagyMiklosZoltan\PycharmProjects\Szakdolgozat2020\RawImages\CIFAR_Images\train'
validation_data_dir = \
    r'C:\Users\NagyMiklosZoltan\PycharmProjects\Szakdolgozat2020\RawImages\CIFAR_Images\validation'

img_width, img_height = 64, 64
im_shape = (img_width, img_height, 3)

batch_size = 256
epochs = 1000
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

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation=relu, input_shape=im_shape))

model.add(Conv2D(filters=16, kernel_size=(3, 3), padding="same"))
model.add(LeakyReLU(0.5))
model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1000, activation=relu))
model.add(Dropout(0.25))
model.add(Dense(1000, activation=relu))
model.add(Dropout(0.25))
model.add(Dense(100, activation=relu))
model.add(Dropout(0.25))
model.add(Dense(100, activation=relu))
model.add(Dropout(0.25))

model.add(Dense(10, activation="softmax"))

model.compile(loss="categorical_crossentropy",
              optimizer=rmsprop(lr=0.0001, decay=1e-6),
              metrics=["acc"])

print(model.summary())

# checkpoint
filepath = "Own_Model_{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='min', period=5)
callbacks_list = [checkpoint]

x = model.fit_generator(generator=train_gen,
                        steps_per_epoch=len(train_gen.filenames) // batch_size,
                        epochs=epochs,
                        validation_data=valid_gen,
                        validation_steps=len(valid_gen.filenames) // batch_size,
                        verbose=1,
                        callbacks=callbacks_list
                        )

plot_history(x)

model.save("OwnModel.hdf5", True)
model.save_weights("OwnModel_Weights.h5")
