import numpy as np
import math
from itertools import combinations_with_replacement


class Average(object):
    def __init__(self, x: int, y: int):
        self.x: float = x
        self.y: float = y
        self.sum: float = 0.0
        self.count: int = 0

    def addToAverage(self, cellvalue: float):
        self.sum = self.sum + cellvalue
        self.count = self.count + 1

    def getCellAverage(self):
        return round(self.sum / self.count, 2)

    def amITheOne(self, x: int, y: int):
        if x == self.x:
            if y == self.y:
                return True
        if y == self.x:
            if x == self.y:
                return True
        return False


def createClassPairs(preds):
    count = np.unique(preds)
    iter = list(combinations_with_replacement(count, 2))
    class_pairs: list= []
    for i in range(len(iter)):
        class_pairs.append(Average(iter[i][0], iter[i][1]))
        print(class_pairs[i].x, end=' ')
        print(class_pairs[i].y)

    return class_pairs


def calculateClassAverages(rdm: np.ndarray, preds):
    cl_rdm = rdm.copy()
    class_pairs = createClassPairs(preds)

    shape = np.shape(rdm)[0]
    """Prepare class average pairs"""
    for i in range(shape):
        for k in range(shape):
            """Find current class_pair"""
            for c in class_pairs:
                "Leaving out diagonal zeros"
                if not i == k:
                    if c.amITheOne(preds[i], preds[k]):
                        c.addToAverage(rdm[i, k])
                        break
    for c in class_pairs:
        print(c.getCellAverage())
    """Create rdm from averages"""
    for i in range(shape):
        for k in range(shape):
            for c in class_pairs:
                if i == k:
                    cl_rdm[i, k] = 0.0
                    break
                else:
                    if c.amITheOne(preds[i], preds[k]):
                        cl_rdm[i, k] = c.getCellAverage()
                        break
    return cl_rdm


