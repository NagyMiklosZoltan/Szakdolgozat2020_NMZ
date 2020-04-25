from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Input, LeakyReLU, UpSampling2D, GlobalAveragePooling2D, \
    BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.applications import ResNet50
import datetime

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

batch_size = 4
num_class = 10


datagen = ImageDataGenerator(rescale=1. / 255)

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

resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

model = Sequential()
model.add(UpSampling2D(input_shape=(img_width, img_height, 3)))
model.add(UpSampling2D())
model.add(UpSampling2D())
model.add(resnet_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(BatchNormalization())
model.add(Dense(num_class, activation='softmax'))

for layer in resnet_model.layers:
    if isinstance(layer, BatchNormalization):
        layer.trainable = True
    else:
        layer.trainable = False

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

# checkpoint
filepath = "cif10_res50_weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
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
    callbacks=callbacks_list
)

print(history.history.keys())
model.save_weights('model_Weights_Cifar10_Resnet50.h5')

end = datetime.datetime.now()
elapsed = end - start
print('Time: ', elapsed)
plot_history(history)
