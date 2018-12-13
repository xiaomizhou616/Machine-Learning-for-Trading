"""
Test a learner.  (c) 2015 Tucker Balch
"""

import numpy as np
import math
import RTLearner as rtl
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

    rmse_in = []
    rmse_out = []
    leaf_size = 1
    leaf_min = 1
    leaf_max = 50
    leaf_array = range(leaf_min,leaf_max+1)
### loop through leaf_size ###

    for leaf_size in range(leaf_min, leaf_max+1):
    # create a learner and train it
        learner = rtl.RTLearner(leaf_size, verbose = False) # create a dtLearner
        learner.addEvidence(trainX, trainY) # train it

        # evaluate in sample
        predY = learner.query(trainX) # get the predictions
        rmse_in = np.append(rmse_in, math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0]))


        # evaluate out of sample
        predY = learner.query(testX) # get the predictions
        rmse_out = np.append(rmse_out, math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0]))

    print leaf_array
    print "In sample results"
    print "RMSE_in: ", rmse_in
    print "Out of sample results"
    print "RMSE_out: ", rmse_out
    plt.figure(figsize=(10,5))
    plt.xlabel("Leaf_size")
    plt.ylabel("RMSE")
    plt.title("Assess overfitting for DTLearner")
    plt.plot(leaf_array, rmse_in, label="In-sample Data")
    plt.plot(leaf_array, rmse_out, label="out-of-sample Data")
    plt.xticks(np.arange(min(leaf_array),max(leaf_array)+1,2.0))
    plt.legend(loc='lower right')
    plt.show()
