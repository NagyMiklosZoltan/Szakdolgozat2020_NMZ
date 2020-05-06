import numpy as np

np.random.seed(1337)  # for reproducibility

from keras import backend as K
from ReStart.Codes.Dataset.MatFilesDataset import *


class DataGenerator(object):
    """docstring for DataGenerator"""

    def __init__(self, x_path: str, y_path: str, batch_size: int):
        self.batch_size = batch_size
        self.images = readMatImages(x_path)
        self.rdm = getAverageEVC_RDM(y_path)
        self.idx_pairs = getIndexPairs(self.rdm.shape[0])
        self.triplets = []
        self.cur_train_index: int = 0
        self.samples_per_train: int = (len(self.idx_pairs) // self.batch_size) * self.batch_size

    def create_triplets_Index(self):
        # create triplets from 2 images and 1 rdm cell data, with pair of indexes, with shuffle
        trips = []
        for i in range(len(self.idx_pairs)):
            x, y = self.idx_pairs[i][0], self.idx_pairs[i][1]
            image_pair = x, y
            trips.append((image_pair, self.rdm[x, y]))
        np.random.shuffle(trips)
        self.triplets = trips.copy()
        self.samples_per_train = len(self.triplets) // self.batch_size

    def create_triplet(self, element):
        x, y = element[0][0], element[0][1]
        image_pair = self.images[x], self.images[y]
        return [image_pair], element[1]

    def create_train_batch(self, min_idx: int, max_idx: int):
        trips = []
        for i in range(min_idx, max_idx):
            one = self.create_triplet(self.triplets[i])
            trips.append(one)
        return trips

    def generator(self):
        """Recreate random shuffled triplets order"""
        self.create_triplets_Index()
        while 1:
            self.cur_train_index += self.batch_size
            if self.cur_train_index >= self.samples_per_train:
                self.cur_train_index = 0
            max_train = self.cur_train_index + self.batch_size
            trips = self.create_train_batch(self.cur_train_index, max_train)
            yield trips


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)



def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return labels[predictions.ravel() < 0.5].mean()
