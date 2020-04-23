import pickle
import numpy as np
from cv2 import imwrite
from os import mkdir
from os.path import isdir

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


meta_input = r'C:\Users\NagyMiklosZoltan\PycharmProjects\Szakdolgozat2020\RawImages\cifar-10-batches-py\batches.meta'
output_path = r'C:\Users\NagyMiklosZoltan\PycharmProjects\Szakdolgozat2020\RawImages\CIFAR_Images'

batch_path =  r'C:\Users\NagyMiklosZoltan\PycharmProjects\Szakdolgozat2020\RawImages\cifar-10-batches-py'

meta = unpickle(meta_input)
class_names = [f.decode("utf-8") for f in meta[b'label_names']]

test_input = r'C:\Users\NagyMiklosZoltan\PycharmProjects\Szakdolgozat2020\RawImages\cifar-10-batches-py\test_batch'


for i in range(1, 2):
    # batch_file = batch_path + '\\data_batch_' + str(i) + ''
    batch_file = test_input
    batch = unpickle(batch_file)
    print("Current Batchdata: " + str(i))
    for k in range(10000):
        arr_1d = np.array(batch[b'data'][k])
        arr_3d = np.reshape(arr_1d, newshape=(3, 32, 32)).transpose()
        label = batch[b'labels'][k]
        curr_class = (meta[b'label_names'][label]).decode('utf-8')
        path = output_path + '\\\\validation\\' + curr_class
        if not isdir(path):
            mkdir(path)
        file = path + '\\' + curr_class + '_' + str(i) + '_' + str(k) + '.jpg'
        imwrite(file, arr_3d)
        if (k % 1000) == 0:
            print(str(i) + ': ' + str(k))






# arr_1d = np.array(dick[b'data'][0])
#
# arr_3d = np.reshape(arr_1d, newshape=(3, 32, 32)).transpose()
# imwrite('CIFAR_Test.jpg', arr_3d)