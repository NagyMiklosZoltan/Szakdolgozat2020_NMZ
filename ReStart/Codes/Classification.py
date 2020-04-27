import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras import optimizers
from keras.layers import Dropout, Flatten, Dense
from keras.utils.np_utils import to_categorical
import datetime

from ReStart.Codes.PlotResults import plot_history
from ReStart.Codes.setKerasSession import setKerasAllow_Groth_lof_device_placement
from ReStart.Codes.directories import train_dir_dict, valid_dir_dict
from ReStart.Codes.preTrain import preTraining

# # pretrain modell with vgg16 predictGenerator
# preTraining()

modelpath = r'C:\Users\NagyMiklosZoltan\PycharmProjects\Szakdolgozat2020\ReStart\Models'
# Create a bottleneck file
top_model_weights_path = modelpath + '\\' + 'bottleneck_fc_model.h5'  # loading up our datasets

train_data_dir = train_dir_dict['base_dir']
validation_data_dir = valid_dir_dict['base_dir']
img_width, img_height = 175, 175
datagen = ImageDataGenerator(rescale=1. / 255)

batch_size = 32

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

print('Meh')

model = Sequential()
model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dense(100))
model.add(keras.layers.LeakyReLU(alpha=0.3))
model.add(Dropout(0.3))
model.add(Dense(50))
model.add(keras.layers.LeakyReLU(alpha=0.3))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

# history = model.fit(train_data, train_labels,
#                     epochs=10,
#                     batch_size=batch_size,
#                     validation_data=(validation_data, validation_labels))


history = model.fit_generator(generator=datagen.flow(train_data, train_labels, batch_size=batch_size),
                              epochs=1000,
                              steps_per_epoch=len(train_data) // batch_size,
                              validation_data=datagen.flow(validation_data, validation_labels, batch_size=batch_size),
                              validation_steps=len(validation_data) // batch_size)


print(history.history.keys())
model.save_weights(top_model_weights_path)

model.save(r'C:\Users\NagyMiklosZoltan\PycharmProjects\Szakdolgozat2020\ReStart\Models\Full_5_ClassModel.h5')

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

