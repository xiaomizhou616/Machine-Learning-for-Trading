"""
Test a learner.  (c) 2015 Tucker Balch
"""

import numpy as np
import math
import pandas as pd
import time
from matplotlib.legend_handler import HandlerLine2D

import LinRegLearner as lrl
import DTLearner as dt
import RTLearner as rt
import sys
import BagLearner as bl
import InsaneLearner as it
import matplotlib.pyplot as plt

if __name__ == "__main__":
    start_time = time.time()
    inf = open('Istanbulcopy4.csv')
    data = np.array([map(float, s.strip().split(',')) for s in inf.readlines()])

    # compute how much of the data is training and testing
    train_rows = int(0.6 * data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    trainX = data[:train_rows, 0:-1]
    trainY = data[:train_rows, -1]
    testX = data[train_rows:, 0:-1]
    testY = data[train_rows:, -1]

    rmseRTArr1 = []  # in sample
    rmseRTArr2 = []  # out of sample
    rmseDTArr1 = []
    rmseDTArr2 = []
    corrRT1 = []  # in sample
    corrRT2 = []  # out of sample
    corrDT1 = []
    corrDT2 = []
    axi = []
    for i in range(1):
        learnerRT = rt.RTLearner(leaf_size=i + 1, verbose=False)
        learnerRT.addEvidence(trainX, trainY)  # training step
        predYRT = learnerRT.query(trainX)  # get the predictions
        rmseBag1 = math.sqrt(((trainY - predYRT) ** 2).sum() / trainY.shape[0])
        cBag = np.corrcoef(predYRT, y=trainY)
        corrRT1.append(cBag[0, 1])
        rmseRTArr1.append(rmseBag1)
        axi.append(i + 1)
        predYBag = learnerRT.query(testX)  # get the predictions
        rmseBag2 = math.sqrt(((testY - predYBag) ** 2).sum() / testY.shape[0])
        rmseRTArr2.append(rmseBag2)
        cRT = np.corrcoef(predYBag, y=testY)
        corrRT2.append(cRT[0, 1])
    print("--- %s seconds ---" % (time.time() - start_time))
