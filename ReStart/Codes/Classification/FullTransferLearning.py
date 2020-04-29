import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.layers import Dropout, Flatten, Dense
from keras.applications import VGG16
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from ReStart.Codes.Visualization.PlotEpoch import PlotLearning

from ReStart.Codes.Dataset.directories import train_dir_dict, valid_dir_dict
from ReStart.Codes.Visualization.PlotResults import plot_history
from ReStart.Codes.Classification.setKerasSession import setKerasAllow_Groth_lof_device_placement


def TrainModelClassification():
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
    # for layer in vgg16.layers:
    #     print(layer.trainable)

    x = vgg16.layers[-1].output
    print(x)
    # x = Conv2D(filters=1024, kernel_size=(5, 5), strides=(2, 2))(x)
    # x = Conv2D(filters=1024, kernel_size=(3, 3), strides=(2, 2))(x)
    # # x = MaxPool2D(pool_size=(5, 5))

    x = Flatten()(x)
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
    # opt = optimizers.SGD(lr=0.01)
    opt = optimizers.RMSprop(lr=0.00001)
    # opt = optimizers.Adam(lr=0.00001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['acc'])

    print(model.summary())

    # checkpoint
    filepath = r"C:\Users\NagyMiklosZoltan\PycharmProjects\Szakdolgozat2020\ReStart\Weights\weights-improvement-{" \
               r"epoch:02d}-{val_loss:.2f}.hdf5 "
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min', period=1)
    callbacks_list = [checkpoint]

    # Realtime epoch acc loss plotting
    plot = PlotLearning()

    history = model.fit_generator(generator=train_gen,
                                  epochs=10,
                                  steps_per_epoch=100,
                                  validation_data=valid_gen,
                                  validation_steps=100,
                                  callbacks=callbacks_list)

    plot_history(history)
    print('Learning finished!')
