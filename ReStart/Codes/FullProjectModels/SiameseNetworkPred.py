from keras.models import Model
from keras.models import load_model
from keras.layers import Dense, Dropout, LeakyReLU, Lambda, Input
from keras import backend as kerasBackEnd

model_path = r'C:\Users\NagyMiklosZoltan\PycharmProjects\Szakdolgozat2020\ReStart\Weights\weights-improvement-02-0.05.hdf5'
my_model = load_model(model_path)

# rewire model
print(my_model.summary())

count = len(my_model.layers)-7

for layer in my_model.layers[:count]:
    layer.trainable = False

x = my_model.layers[-7].output
x = Dense(100, name='Dense_new_1')(x)
x = LeakyReLU(alpha=0.3, name='LeakyRelu_new_1')(x)
x = Dropout(0.25, name='Dropout_new_1')(x)
x = Dense(100, name='Dense_new_2')(x)
x = LeakyReLU(alpha=0.3, name='LeakyRelu_new_2')(x)
x = Dropout(0.25, name='Dropout_new_2')(x)
x = Dense(25, activation='sigmoid', name='Dense_new_3')(x)
# x = LeakyReLU(alpha=0.3, name='LeakyRelu_new_3')(x)

model = Model(inputs=my_model.inputs, outputs=x)

left_input = Input(shape=(175, 175, 3))
right_input = Input(shape=(175, 175, 3))


left_wing = model(left_input)
right_wing = model(right_input)

L1_layer = Lambda(lambda tensors: kerasBackEnd.abs(tensors[0] - tensors[1]))
L1_distance = L1_layer([left_wing, right_wing])

last_layer = Dense(1, activation='sigmoid')(L1_distance)

siameseNetwork = Model(inputs=[left_input, right_input], outputs=last_layer)


