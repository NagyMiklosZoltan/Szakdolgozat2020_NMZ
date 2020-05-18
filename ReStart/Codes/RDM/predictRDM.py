from keras.models import load_model
from ReStart.Codes.FullProjectModels.Generator import DataGenerator
import keras

root = r'C:\Users\NagyMiklosZoltan\PycharmProjects\Szakdolgozat2020'
trainY_path = root + r'\algonautsChallenge2019\Training_Data\92_Image_Set\target_fmri.mat'


def getPrediction(new_model : bool):
    if new_model:
        my_model = load_model(
            r'C:\Users\NagyMiklosZoltan\PycharmProjects\Szakdolgozat2020\ReStart\Codes\FullProjectModels\Siamese_weights-improvement-08-0.04.hdf5',
            custom_objects={'keras': keras})
    else:
        my_model = load_model(
            r'C:\Users\NagyMiklosZoltan\PycharmProjects\Szakdolgozat2020\ReStart\Codes\FullProjectModels\SiameseSave.hdf5',
            custom_objects={'keras': keras})

    trainX_path = root + r'\algonautsChallenge2019\Training_Data\92_Image_Set\92images.mat'

    test_gen = DataGenerator(x_path=trainX_path,
                             y_path=trainY_path,
                             batch_size=1)

    preds = my_model.predict_generator(test_gen.generator(False), steps=len(test_gen.idx_pairs))

    preds: list = [f[0] for f in preds]
    # print('Pred length:\t', end='')
    # print(len(preds))
    # print('Predictions:\t', end='')
    # print(preds)

    return preds




