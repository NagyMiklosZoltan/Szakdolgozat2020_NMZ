from keras.models import load_model
from ReStart.Codes.FullProjectModels.Generator import DataGenerator
import keras
from keras import backend as kerasBackEnd

my_model = load_model(
    r'C:\Users\NagyMiklosZoltan\PycharmProjects\Szakdolgozat2020\ReStart\Codes\FullProjectModels\SiameseSave.hdf5')

root = r'C:\Users\NagyMiklosZoltan\PycharmProjects\Szakdolgozat2020'

trainX_path = root + r'\algonautsChallenge2019\Training_Data\92_Image_Set\92images.mat'
trainY_path = root + r'\algonautsChallenge2019\Training_Data\92_Image_Set\target_fmri.mat'

test_gen = DataGenerator(x_path=trainX_path,
                         y_path=trainY_path,
                         batch_size=1)

preds = my_model.predict_generator(test_gen.generator(False), steps=test_gen.samples_per_train)
print(preds)
