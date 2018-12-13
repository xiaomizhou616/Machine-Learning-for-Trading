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
    '''
    if len(sys.argv) != 2:
        print "Usage: python testlearner.py <filename>"
        sys.exit(1)
    '''
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

    # create a learner and train it
    '''
    learner = lrl.LinRegLearner(verbose = True) # create a LinRegLearner
    learner.addEvidence(trainX, trainY) # train it
    print learner.author()
    '''
    rmseRTArr1 = []  # in sample
    rmseRTArr2 = []  # out of sample
    rmseDTArr1 = []
    rmseDTArr2 = []
    corrRT1 = []  # in sample
    corrRT2 = []  # out of sample
    corrDT1 = []
    corrDT2 = []
    axi = []
    for i in range(40):
        learnerRT = rt.RTLearner(leaf_size=i + 1, verbose=False)
        learnerDT = dt.DTLearner(leaf_size=i + 1, verbose=False)  # constructor
        learnerRT.addEvidence(trainX, trainY)  # training step
        learnerDT.addEvidence(trainX, trainY)
        predYRT = learnerRT.query(trainX)  # get the predictions
        predYDT = learnerDT.query(trainX)
        rmseBag1 = math.sqrt(((trainY - predYRT) ** 2).sum() / trainY.shape[0])
        rmseDT1 = math.sqrt(((trainY - predYDT) ** 2).sum() / trainY.shape[0])
        cBag = np.corrcoef(predYRT, y=trainY)
        cDT = np.corrcoef(predYDT, y=trainY)
        corrRT1.append(cBag[0, 1])
        corrDT1.append(cDT[0, 1])
        rmseRTArr1.append(rmseBag1)
        rmseDTArr1.append(rmseDT1)
        axi.append(i + 1)
        # evaluate out of sample
        predYBag = learnerRT.query(testX)  # get the predictions
        predYDT = learnerDT.query(testX)
        rmseBag2 = math.sqrt(((testY - predYBag) ** 2).sum() / testY.shape[0])
        rmseDT2 = math.sqrt(((testY - predYDT) ** 2).sum() / testY.shape[0])
        rmseRTArr2.append(rmseBag2)
        rmseDTArr2.append(rmseDT2)
        cRT = np.corrcoef(predYBag, y=testY)
        cDT = np.corrcoef(predYDT, y=testY)
        corrRT2.append(cRT[0, 1])
        corrDT2.append(cDT[0, 1])
        print("end", i + 1)

    line1, = plt.plot(axi, rmseRTArr1, label="In sample(RT)")
    line2, = plt.plot(axi, rmseRTArr2, label="Out of Sample(RT)")
    line3, = plt.plot(axi, rmseDTArr1, label="In sample(DT)")
    line4, = plt.plot(axi, rmseDTArr2, label="Out of Sample(DT)")
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=5)})
    plt.xlabel('Leaf size')
    plt.ylabel('RMSE')
    plt.title("RTLearner VS DTLearner RMSE Analysis")
    plt.grid(True)
    plt.show()
    print("end")


    '''
    line1, = plt.plot(axi, corrRT1, label="In sample(RT)")
    line2, = plt.plot(axi, corrRT2, label="Out of Sample(RT)")
    line3, = plt.plot(axi, corrDT1, label="In sample(DT)")
    line4, = plt.plot(axi, corrDT2, label="Out of Sample(DT)")
    

    '''

    '''
    learner = rt.RTLearner(leaf_size = 1, verbose = False) # constructor
    learner.addEvidence(trainX, trainY) # training step
    print learner.author()

    '''

    '''
    learner = bl.BagLearner(learner=dt.DTLearner, kwargs={"leaf_size": 1}, bags=20, boost=False, verbose=False)     
    learner.addEvidence(trainX, trainY)                                                                             
    '''

    '''
        learner = it.InsaneLearner(verbose = False) # constructor                
        learner.addEvidence(trainX, trainY) # training step# evaluate in sample  
    '''





    # evaluate in sample
