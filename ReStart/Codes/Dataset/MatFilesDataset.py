import h5py
import numpy as np
import scipy.io as ScIO
from itertools import combinations


def getAverageEVC_RDM(input):
    fMRI_mat = input
    fMRI = h5py.File(fMRI_mat, 'r')
    # print(fMRI.keys())

    data = fMRI.get('EVC_RDMs')
    data = np.array(data)

    # print(np.shape(data))
    average = np.mean(data, axis=2)
    # print(np.shape(average))
    return average


def readMatImages(input):
    im_mat = input
    imgs = ScIO.loadmat(im_mat)
    # print(imgs.keys())
    # print(imgs['visual_stimuli'])
    arr = np.array(imgs['visual_stimuli'])
    # print(np.shape(arr))
    arr = arr[0]['pixels']
    arr /= 255
    return arr


def getIndexPairs(count):
    indexes = [i for i in range(count)]
    index_pairs = combinations(indexes, 2)
    return list(index_pairs)

a = getAverageEVC_RDM(r'C:\Users\NagyMiklosZoltan\PycharmProjects\Szakdolgozat2020\algonautsChallenge2019'
                      r'\Training_Data\92_Image_Set\target_fmri.mat')
print(np.shape(a))


b = readMatImages(r'C:\Users\NagyMiklosZoltan\PycharmProjects\Szakdolgozat2020\algonautsChallenge2019\Training_Data'
                  r'\92_Image_Set\92images.mat')
print(b.dtype)
print(np.shape(b[0]))
print(len(getIndexPairs(92)))



