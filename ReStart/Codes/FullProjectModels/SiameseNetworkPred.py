from keras.models import Model
from keras.models import load_model
from keras.layers import Dense, Dropout, LeakyReLU, Lambda, Input
import keras.backend
from keras.optimizers import Adam, Adagrad
from keras.losses import MSE

# own generator
from ReStart.Codes.FullProjectModels.Generator import DataGenerator

root = r'C:\Users\NagyMiklosZoltan\PycharmProjects\Szakdolgozat2020'


trainX_path = root + r'\algonautsChallenge2019\Training_Data\92_Image_Set\92images.mat'
trainY_path = root + r'\algonautsChallenge2019/Training_Data/92_Image_Set/target_fmri.mat'
validX_path = root + r'\algonautsChallenge2019/Training_Data/118_Image_Set/118images.mat'
validY_path = root + r'\algonautsChallenge2019/Training_Data/118_Image_Set/target_fmri.mat'
batch_size = 16

train_gen = DataGenerator(x_path=trainX_path,
                          y_path=trainY_path,
                          batch_size=batch_size)

valid_gen = DataGenerator(x_path=validX_path,
                          y_path=validY_path,
                          batch_size=batch_size)

model_path = root + r'\ReStart\Weights\weights-improvement-02-0.05.hdf5'
my_model = load_model(model_path)

# rewire model
print(my_model.summary())

count = len(my_model.layers)-7

# az átvett régtegek befagyasztása
for layer in my_model.layers[:count]:
    layer.trainable = False

# Model kiegészítés és utolsó rétegek cseréje
x = my_model.layers[-7].output
x = Dense(100, name='Dense_new_1')(x)
x = LeakyReLU(alpha=0.3, name='LeakyRelu_new_1')(x)
x = Dropout(0.4, name='Dropout_new_1')(x)
x = Dense(100, name='Dense_new_2')(x)
x = LeakyReLU(alpha=0.3, name='LeakyRelu_new_2')(x)
x = Dropout(0.3, name='Dropout_new_2')(x)
x = Dense(25, activation='sigmoid', name='Dense_new_3')(x)
# x = LeakyReLU(alpha=0.3, name='LeakyRelu_new_3')(x)

model = Model(inputs=my_model.inputs, outputs=x)

left_input = Input(shape=(175, 175, 3))
right_input = Input(shape=(175, 175, 3))


left_wing = model(left_input)
right_wing = model(right_input)

L1_layer = Lambda(lambda tensors: keras.backend.abs(tensors[0] - tensors[1]))
L1_distance = L1_layer([left_wing, right_wing])

last_layer = Dense(1, activation='sigmoid')(L1_distance)

siameseNetwork = Model(inputs=[left_input, right_input], outputs=last_layer)

opt = Adagrad()
# opt = Adam(lr=0.005)

siameseNetwork.compile(optimizer=opt,
                       loss=MSE)

steps_per_epoch = train_gen.samples_per_train // batch_size
print(steps_per_epoch)


valid_steps = valid_gen.samples_per_train // batch_size
history = siameseNetwork.fit_generator(generator=train_gen.generator(True),
                                       epochs=1,
                                       steps_per_epoch=steps_per_epoch,
                                       validation_data=valid_gen.generator(True),
                                       validation_steps=valid_steps)

siameseNetwork.save('SiameseSave.hdf5')

preds = siameseNetwork.predict_generator(train_gen.generator(False), steps=train_gen.samples_per_train)
print(preds)
print(preds.max())
print(preds.min())