"""
Test a learner.  (c) 2015 Tucker Balch
"""

import numpy as np
import math
import pandas as pd
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
    rmseBagArr1 = []  # in sample
    rmseBagArr2 = []  # out of sample
    rmseRTArr1 = []
    rmseRTArr2 = []
    corrBag1 = []  # in sample
    corrBag2 = []  # out of sample
    corrRT1 = []
    corrRT2 = []
    axi = []
    for i in range(100):
        learnerBag = bl.BagLearner(learner=dt.DTLearner, kwargs={"leaf_size": i + 1}, bags=10, boost=False, verbose=False)
        learnerRT = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": i + 1}, bags=10, boost=False, verbose=False)  # constructor
        learnerBag.addEvidence(trainX, trainY)  # training step
        learnerRT.addEvidence(trainX, trainY)
        predYBag = learnerBag.query(trainX)  # get the predictions
        predYRT = learnerRT.query(trainX)
        rmseBag1 = math.sqrt(((trainY - predYBag) ** 2).sum() / trainY.shape[0])
        rmseRT1 = math.sqrt(((trainY - predYRT) ** 2).sum() / trainY.shape[0])
        cBag = np.corrcoef(predYBag, y=trainY)
        cRT = np.corrcoef(predYRT, y=trainY)
        corrBag1.append(cBag[0, 1])
        corrRT1.append(cRT[0, 1])
        rmseBagArr1.append(rmseBag1)
        rmseRTArr1.append(rmseRT1)
        axi.append(i + 1)
        # evaluate out of sample
        predYBag = learnerBag.query(testX)  # get the predictions
        predYDT = learnerRT.query(testX)
        rmseBag2 = math.sqrt(((testY - predYBag) ** 2).sum() / testY.shape[0])
        rmseRT2 = math.sqrt(((testY - predYDT) ** 2).sum() / testY.shape[0])
        rmseBagArr2.append(rmseBag2)
        rmseRTArr2.append(rmseRT2)
        cBag = np.corrcoef(predYBag, y=testY)
        cRT = np.corrcoef(predYDT, y=testY)
        corrBag2.append(cBag[0, 1])
        corrRT2.append(cRT[0, 1])
        print("end", i + 1)

    line1, = plt.plot(axi, rmseBagArr1, label="In sample(DTBag)")
    line2, = plt.plot(axi, rmseBagArr2, label="Out of Sample(DTBag)")
    line3, = plt.plot(axi, rmseRTArr1, label="In sample(RTBag)")
    line4, = plt.plot(axi, rmseRTArr2, label="Out of Sample(RTBag)")
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=1)})
    plt.yticks(np.arange(0, 0.008, 0.002))
    plt.xlabel('Leaf size')
    plt.ylabel('Error')
    plt.title("DTBagLearner VS RTBagLearner RMSE Analysis")
    plt.grid(True)
    plt.show()




    print("end")

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
