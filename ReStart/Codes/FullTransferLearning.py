import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras import optimizers
from keras.layers import Dropout, Flatten, Dense
from keras.applications import VGG16
from keras.callbacks import ModelCheckpoint
from keras.models import Model


from ReStart.Codes.directories import train_dir_dict, valid_dir_dict
from ReStart.Codes.PlotResults import plot_history
from ReStart.Codes.setKerasSession import setKerasAllow_Groth_lof_device_placement

setKerasAllow_Groth_lof_device_placement()

train_data_dir = train_dir_dict['base_dir']
validation_data_dir = valid_dir_dict['base_dir']
img_width, img_height = 175, 175
datagen = ImageDataGenerator(rescale=1. / 255)

batch_size = 32

train_gen = datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

valid_gen = datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

shape = (175, 175, 3)

vgg16 = VGG16(include_top=False, weights='imagenet', input_shape=shape)

# 10-> layer block3_pool (MaxPooling2D)   (None, 21, 21, 256)
for layer in vgg16.layers[:10]:
    layer.trainable = False
for layer in vgg16.layers:
    print(layer.trainable)

x = Flatten()(vgg16.layers[-1].output)
x = Dense(256)(x)
x = keras.layers.LeakyReLU(alpha=0.3)(x)
x = Dropout(0.25)(x)
x = Dense(256)(x)
x = keras.layers.LeakyReLU(alpha=0.3)(x)
x = Dropout(0.25)(x)
x = Dense(50)(x)
x = keras.layers.LeakyReLU(alpha=0.3)(x)
x = Dropout(0.25)(x)
x = Dense(5, activation='softmax')(x)


model = Model(inputs=vgg16.inputs, outputs=x)

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=0.001, decay=1e-6),
              metrics=['acc'])

print(model.summary())

# checkpoint
filepath = r"C:\Users\NagyMiklosZoltan\PycharmProjects\Szakdolgozat2020\ReStart\Weights\weights-improvement-{" \
           r"epoch:02d}-{val_loss:.2f}.hdf5 "
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min', period=1)
callbacks_list = [checkpoint]

history = model.fit_generator(generator=train_gen,
                              epochs=1000,
                              steps_per_epoch=len(train_gen.filenames) // batch_size,
                              validation_data=valid_gen,
                              validation_steps=len(valid_gen.filenames) // batch_size,
                              callbacks=callbacks_list)

plot_history(history)
