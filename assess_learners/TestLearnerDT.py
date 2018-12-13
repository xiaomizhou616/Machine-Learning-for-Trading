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
    inf = open('Istanbul.csv')
    data = np.array([map(float, s.strip().split(',')) for s in inf.readlines()])
    np.random.shuffle(data)

    # compute how much of the data is training and testing
    train_rows = int(0.6 * data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    trainX = data[:train_rows, 0:-1]
    trainY = data[:train_rows, -1]
    testX = data[train_rows:, 0:-1]
    testY = data[train_rows:, -1]
    rmseDTArr1 = []
    rmseDTArr2 = []
    corrDT1 = []
    corrDT2 = []
    axi = []
    for i in range(100):
        learnerDT = dt.DTLearner(leaf_size=i + 1, verbose=False)  # constructor
        learnerDT.addEvidence(trainX, trainY)
        predYDT = learnerDT.query(trainX)
        rmseDT1 = math.sqrt(((trainY - predYDT) ** 2).sum() / trainY.shape[0])
        cDT = np.corrcoef(predYDT, y=trainY)
        corrDT1.append(cDT[0, 1])
        rmseDTArr1.append(rmseDT1)
        axi.append(i + 1)
        # evaluate out of sample
        predYDT = learnerDT.query(testX)
        rmseDT2 = math.sqrt(((testY - predYDT) ** 2).sum() / testY.shape[0])
        rmseDTArr2.append(rmseDT2)
        cDT = np.corrcoef(predYDT, y=testY)
        corrDT2.append(cDT[0, 1])
        print("end", i+1)
    line1, = plt.plot(axi, rmseDTArr1, label="In sample(DT)")
    line2, = plt.plot(axi, rmseDTArr2, label="Out of Sample(DT)")
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=1)})
    plt.yticks(np.arange(0, 0.01, 0.002))
    plt.xlabel('Leaf size')
    plt.ylabel('RMSE')
    plt.title("DTLearner RMSE Analysis")
    plt.grid(True)
    plt.show()