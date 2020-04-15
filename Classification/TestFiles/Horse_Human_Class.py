import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications

from DataPreWork.TestFiles.horse_prework import preworkAllDirs
from Classification.TestFiles.PlotResults import plot_history

# preworkAllDirs()
print("*****************************************")
print("PreWork Done!")
print("*****************************************")

# dimensions of our images.
img_width, img_height = 175, 175

train_data_dir = \
    'C:\\Users\\NagyMiklosZoltan\\PycharmProjects\\Szakdolgozat2020\\RawImages\\horse-or-human\\train'
validation_data_dir = \
    'C:\\Users\\NagyMiklosZoltan\\PycharmProjects\\Szakdolgozat2020\\RawImages\\horse-or-human\\validation'

data_gen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1. / 255)

batch_size = 8
epochs = 100

train_gen = data_gen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,
    color_mode="rgb")

valid_gen = data_gen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,
    color_mode="rgb")

vgg16 = applications.VGG16(
    input_shape=(img_width, img_height, 3),
    include_top=False
)

for layer in vgg16.layers[:10]:
    layer.trainable = False

model = Sequential([
    vgg16,
    Flatten(),
    Dense(2048, activation='relu'),
    Dropout(0.25),
    Dense(512, activation='relu'),
    Dropout(0.25),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy')

print(model.summary())

x = model.fit_generator(generator=train_gen,
                        epochs=epochs,
                        steps_per_epoch=batch_size,
                        validation_data=valid_gen,
                        validation_steps=(len(valid_gen) // batch_size),
                        verbose=2
                        )
print(x.history.keys())
plot_history(x)
