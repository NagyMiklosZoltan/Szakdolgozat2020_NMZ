import numpy as np
from numpy.random import randint
from scipy.stats import spearmanr
from ReStart.Codes.RDM.predictRDM import getPrediction, root, trainY_path
from ReStart.Codes.Dataset.MatFilesDataset import getAverageEVC_RDM, getFirstRealEVC_RDM

count = 10


def generateTestData():
    A = [26, 18, 58, 99, 15, 64, 92, 85, 27, 84]
    B = np.empty(count, int)
    for i in range(count):
        if A[i] < 20:
            B[i] = A[i] + (randint(0, 20))
        elif A[i] > 90:
            B[i] = A[i] + (randint(-20, 0))
        else:
            B[i] = A[i] + (randint(-10, 10))
    return A, B


def convertPredsToFull(preds: list):
    count = 0
    full = np.empty((92, 92), float)
    for i in range(92):
        for k in range(i, 92):
            if i == k:
                full[i, k] = 0.0
            else:
                full[i, k] = preds[count]
                full[k, i] = preds[count]
            count = count + 1

    print('Full:', end='')
    print(len(full))
    return full


def calculateRS(A, B):
    # R Squared calculation
    A_mean = np.mean(A)

    # Actual values to A_mean
    A_dif = A - A_mean
    A_dif_squared = np.square(A_dif)

    # Sum of Squared A_Dif values
    A_ds_sum = np.sum(A_dif_squared)
    print('A_ds_sum:\t', end='')
    print(A_ds_sum)

    # Estimated values to A_mean
    B_dif = B - A_mean
    B_dif_squared = np.square(B_dif)

    # Sum of Squared A_Dif values
    B_ds_sum = np.sum(B_dif_squared)
    print('B_ds_sum:\t', end='')
    print(B_ds_sum)

    R_squared = 1 - B_ds_sum / A_ds_sum
    print('R_Squared:\t', end='')
    print(R_squared)

    subs = []
    for i in range(len(A)):
        subs.append(np.abs(A[i] - B[i]))

    score = np.mean(subs) * 100
    print('Score: ', end='')
    print(score)


# **********************************************************************************************************************
# SCORE CALCULATION
# **********************************************************************************************************************

actual_Data = getAverageEVC_RDM(trainY_path)
# actual_Data = getFirstRealEVC_RDM(trainY_path)
preds = getPrediction()

pred_Data = convertPredsToFull(preds)

# print(actual_Data)
# print(pred_Data)

calculateRS(actual_Data, pred_Data)
