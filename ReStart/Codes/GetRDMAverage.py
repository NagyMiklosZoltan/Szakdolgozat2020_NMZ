import numpy as np
import math


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
        self.sum += n
        self.count += 1

    def getAverage(self):
        avg = self.sum / self.count
        return round(avg, 2)

    def itsTheOne(self, n1, n2):
        if n1 != n2 \
                and (n1 in self.pair) \
                and (n2 in self.pair):
            return True
        return False


def createClassPairs():
    pairs_ = []
    for i in range(10):
        x = ClassPair()
        pairs_.append(x)

    pairs_[0].init(1, 2)
    pairs_[1].init(1, 3)
    pairs_[2].init(1, 4)
    pairs_[3].init(1, 5)
    pairs_[4].init(2, 3)
    pairs_[5].init(2, 4)
    pairs_[6].init(2, 5)
    pairs_[7].init(3, 4)
    pairs_[8].init(3, 5)
    pairs_[9].init(4, 5)

    return pairs_


def chosePairIndex(pairs_, n1_: int, n2_: int):
    for idx in range(np.shape(pairs_)[0]):
        if pairs_[idx].itsTheOne(n1=n1_ + 1, n2=n2_ + 1):
            return idx
    return -1


# ********************** PROGRAM ***********************

arr = createTestRDMs(3, 8)
prediction = shittyModelPrediction('Képeket átadjuk!')

pairs = createClassPairs()

for f in range(arr.shape[0]):
    for i in range(arr.shape[1]):
        for k in range(arr.shape[2]):
            idx = chosePairIndex(pairs, i, k)
            if not idx < 0:
                pairs[idx].addToSum(arr[f, i, k])

new_arr = np.empty(shape=(arr.shape[1], arr.shape[2]), dtype=float)
for i in range(new_arr.shape[0]):
    for k in range(new_arr.shape[1]):
        idx = chosePairIndex(pairs, i, k)
        if idx >= 0:
            new_arr[i, k] = pairs[idx].getAverage()

for i in range(new_arr.shape[0]):
    for k in range(new_arr.shape[1]):
        data = "{:.2f}".format(new_arr[i, k])
        print(data.ljust(4, '0'), end='\t')
    print()
