import numpy as np
import math
import itertools

# CREATE RANDOM TESTARRAY
def createTestRDMs(z, im_num):
    x, y = im_num, im_num
    arr_ = np.random.rand(z, x, y)
    for i_ in range(arr_.shape[0]):
        for k_ in range(arr_.shape[1]):
            for f_ in range(arr_.shape[2]):
                if k_ == f_:
                    arr_[i_, k_, f_] = 0.0
                else:
                    num = arr_[i_, k_, f_] * 100
                    arr_[i_, k_, f_] = math.ceil(num) / 100

    return arr_


# Shitty model 5class prediction for the images
def shittyModelPrediction(img_array):
    low: int = 0
    high: int = 5
    prediction_ = np.ceil(np.random.uniform(low=low, high=high, size=92))
    return prediction_


class ClassPair:
    pair: list
    count: int
    sum: float

    def init(self, c_1, c_2):
        self.count = 0
        self.sum = 0.0
        self.pair = [c_1, c_2]

    def addToSum(self, n):
        self.sum = self.sum + 1.0
        self.count = self.count + 1

    def getAverage(self):
        avg = self.sum / float(self.count)
        return round(avg, 2)

    def itsTheOne(self, n1, n2):
        if n1 != n2 \
                and (n1 in self.pair) \
                and (n2 in self.pair):
            return True
        return False

    def __str__(self):
        return 'pár típus:' + str(self.pair)


def createClassPairs(num_class):
    numbers = [f for f in range(0, num_class)]
    print(numbers)
    combs = list(itertools.combinations(numbers, 2))

    pairs_ = []
    for i in range(10):
        x = ClassPair()
        x.init(c_1=combs[i][0], c_2=combs[i][1])
        pairs_.append(x)

    return pairs_


def chosePairIndex(pairs_, n1_: int, n2_: int):
    for idx in range(np.shape(pairs_)[0]):
        if pairs_[idx].itsTheOne(n1=math.ceil(n1_), n2=math.ceil(n2_)):
            return idx
    return -1

def GetSampleRDMAverages(sample, pairs, predictions):
    counter = 0
    for f in range(sample.shape[0]):
        for i in range(sample.shape[1]):
            for k in range(sample.shape[2]):
                num_1 = predictions[i]
                num_2 = predictions[k]
                idx = chosePairIndex(pairs, num_1, num_2)
                if not idx < 0:
                    pairs[idx].addToSum(sample[f, i, k])
                else:
                    counter = counter + 1
                    print((num_1, num_2, counter))



    for f in pairs:
        if (f.count <= 0):
            print(f)

    # Createing RDM of Averages
    new_arr = np.zeros(shape=(sample.shape[1], sample.shape[2]))
    for i in range(new_arr.shape[0]):
        for k in range(new_arr.shape[1]):
            num_1 = predictions[i]
            num_2 = predictions[k]
            idx = chosePairIndex(pairs, num_1, num_2)
            if idx >= 0:
                new_arr[i, k] = pairs[idx].getAverage()

    return new_arr


# ********************** PROGRAM ***********************

arr = createTestRDMs(3, 8)
prediction = shittyModelPrediction('Képeket átadjuk!')
num_class = len(np.unique(prediction))
pairs = createClassPairs(num_class)

new_arr = GetSampleRDMAverages(arr, pairs, prediction)



# for i in range(new_arr.shape[0]):
#     for k in range(new_arr.shape[1]):
#         data = "{:.2f}".format(new_arr[i, k])
#         print(data.ljust(4, '0'), end='\t')
#     print()
