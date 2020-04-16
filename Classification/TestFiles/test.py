# from keras.preprocessing.image import ImageDataGenerator
# from keras.models import Sequential
# from keras.layers import Dropout, Flatten, Dense
# from keras import applications
import tensorflow as tf
print(tf.__version__)



# USEFULL Things

    # ImageDataGenerator
    # rotation_range is a value in degrees (0-180), a range within which to randomly rotate pictures
    # width_shift and height_shift are ranges (as a fraction of total width or height) within which to randomly translate pictures vertically or horizontally
    # rescale is a value by which we will multiply the data before any other processing. Our original images consist in RGB coefficients in the 0-255,
    # but such values would be too high for our models to process (given a typical learning rate), so we target values between 0 and 1 instead by scaling with a 1/255. factor.
    # horizontal_flip is for randomly flipping half of the images horizontally --relevant when there are no assumptions of horizontal assymetry (e.g. real-world pictures).
    # fill_mode is the strategy used for filling in newly created pixels, which can appear after a rotation or a width/height shift.
    # cval: Float or Int. Value used for points outside the boundaries when fill_mode = "constant".

# # predict the probability across all output classes
# yhat = model.predict(image)
# # convert the probabilities to class labels
# label = decode_predictions(yhat)
