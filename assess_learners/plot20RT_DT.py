"""
Test a learner.  (c) 2015 Tucker Balch
"""
import time
import numpy as np
import math
import BagLearner as bl
import RTLearner as rtl
import DTLearner as dtl
import sys
import matplotlib.pyplot as plt

if __name__=="__main__":
    if len(sys.argv) != 2:
        print "Usage: python testlearner.py <filename>"
        sys.exit(1)
    inf = open(sys.argv[1])
    data = np.array([map(float,s.strip().split(',')) for s in inf.readlines()])

    # compute how much of the data is training and testing
    train_rows = int(0.6* data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    trainX = data[:train_rows,0:-1]
    trainY = data[:train_rows,-1]
    testX = data[train_rows:,0:-1]
    testY = data[train_rows:,-1]

    print testX.shape
    print testY.shape

    leaf_size = 1
    leaf_min = 1
    leaf_max = 50
    leaf_array = range(leaf_min,leaf_max+1)
    rmse_inRT = []
    rmse_outRT = []

    rmse_inDT = []
    rmse_outDT = []
    total_bags= 5
### loop through leaf_size ###

    start1 = time.time()
    for leaf_size in range(leaf_min, leaf_max+1):
    # create a learner and train it
        learner = bl.BagLearner(learner=rtl.RTLearner, kwargs={"leaf_size":1}, bags=total_bags,boost=False,verbose=False) # create a Bag of 20 rtLearner
        learner.addEvidence(trainX, trainY) # train it

        # evaluate in sample
        predY = learner.query(trainX) # get the predictions
        rmse_inRT = np.append(rmse_inRT, math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0]))

        # evaluate out of sample
        predY = learner.query(testX) # get the predictions
        rmse_outRT = np.append(rmse_outRT, math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0]))
    end1 = time.time()
    print total_bags
    print " Bags, time:", end1-start1

    start2 = time.time()
    for leaf_size in range(leaf_min, leaf_max+1):
    # create a learner and train it
        learner = dtl.DTLearner(leaf_size, verbose = False) # create a dtLearner
        learner.addEvidence(trainX, trainY) # train it

        # evaluate in sample
        predY = learner.query(trainX) # get the predictions
        rmse_inDT = np.append(rmse_inDT, math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0]))

        # evaluate out of sample
        predY = learner.query(testX) # get the predictions
        rmse_outDT = np.append(rmse_outDT, math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0]))
    end2 = time.time()
    print "DT learner, time:", end2-start2
    plt.figure(figsize=(10,5))

    plt.xlabel("leaf_size")
    plt.ylabel("RMSE")
    plt.title("Compare DT and a bag of 5RT")
    plt.plot(leaf_array, rmse_inRT, label="Bag of 5RT: In-sample Data")
    plt.plot(leaf_array, rmse_outRT, label="Bag of 5RT: out-of-sample Data")

    plt.plot(leaf_array, rmse_inDT, label="DT: In-sample Data")
    plt.plot(leaf_array, rmse_outDT, label="DT: out-of-sample Data")


    plt.xticks(np.arange(min(leaf_array),max(leaf_array)+1,2.0))
    plt.legend(loc='lower right')
    plt.savefig("5RT_DT")
    #plt.show()
