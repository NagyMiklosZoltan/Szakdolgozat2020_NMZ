import numpy as np
np.random.seed(1337)  # for reproducibility

import random
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda
from keras import backend as K
from ReStart.Codes.Dataset.MatFilesDataset import *

class DataGenerator(str, str, int):
    """docstring for DataGenerator"""
    def __init__(self, x_path: str, y_path: str, batch_size: int):
        self.batch_size = batch_size
        self.images = readMatImages(x_path)
        self.rdm = getAverageEVC_RDM(y_path)
        self.idx_pairs = getIndexPairs(self.rdm.shape[0])
        self.triplets = []
        self.cur_train_index = 0
        self.samples_per_train = 0

    def create_triplets_Index(self):
        # create triplets from 2 images and 1 rdm cell data, with pair of indexes, with shuffle
        trips = []
        for i in range(len(self.idx_pairs)):
            x, y = self.idx_pairs[i][0], self.idx_pairs[i][1]
            image_pair = x, y
            trips.append((image_pair, self.rdm[x, y]))
        np.random.shuffle(trips)
        self.triplets = trips.copy()
        self.samples_per_train = len(self.triplets)//self.batch_size

    def create_triplet_Train(self, element):
        x, y = element[0], element[1]
        image_pair = self.images[x], self.images[y]
        return [image_pair], element[2]

    def generator(self):
        """Recreate random shuffled triplets order"""
        self.create_triplets_Index()
        while 1:
            self.cur_train_index += self.batch_size
            if self.cur_train_index >= self.samples_per_train:
                self.cur_train_index = 0

            trip = self.create_triplet_Train(element=())
            yield




def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return labels[predictions.ravel() < 0.5].mean()





