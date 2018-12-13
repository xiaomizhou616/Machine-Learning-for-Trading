"""
Test a learner.  (c) 2015 Tucker Balch
"""

__author__ = 'swang637'

import math
import numpy as np
import BagLearner as bg
import DTLearner as dtl
import matplotlib.pyplot as plt
import pandas as pd
import sys

if __name__=="__main__":
    if len(sys.argv) != 2:
        print "Usage: python testlearner.py <filename>"
        sys.exit(1)
    inf = open(sys.argv[1])
    data = np.array([map(float,s.strip().split(',')) for s in inf.readlines()])

    train_rows = int(0.6* data.shape[0])
    test_rows = data.shape[0] - train_rows

    trainX = data[:train_rows,0:-1]
    trainY = data[:train_rows,-1]
    testX = data[train_rows:,0:-1]
    testY = data[train_rows:,-1]
    print testX.shape
    print testY.shape


    leaf_size = 10
    rmse_inDT = []
    rmse_outDT = []
    total_bags = 20
    bag_array = range(1,total_bags+1)


    for i in range(1,total_bags+1):
        learner = bg.BagLearner(learner = dtl.DTLearner,kwargs={"leaf_size":10}, bags = i, boost = False,verbose=False)
        learner.addEvidence(trainX,trainY)

        #evaluate in samples
        predY=learner.query(trainX)
        rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
        rmse_inDT = np.append(rmse_inDT,rmse)

        #evalueat out of sample
        predY = learner.query(testX)
        rmse_outDT = np.append(rmse_outDT, math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0]))

    plt.figure(figsize=(10,5))
    plt.xlabel("bag_size")
    plt.ylabel("RMSE")
    plt.title("Overfitting and bagging(LeafSize10)")
    plt.ylim(0,0.01)
    plt.plot(bag_array, rmse_inDT, label=" In-sample Data")
    plt.plot(bag_array, rmse_outDT, label="out-of-sample Data")



    plt.xticks(np.arange(min(bag_array),max(bag_array)+1,2.0))
    plt.legend(loc='lower right')
    plt.savefig("20bagging_leafsize10")
    plt.show()

'''
    np1_test=np.array(rmse_Bag_test)
    np2_test=np.array(c_Bag_test)
    df2=pd.DataFrame(np1_test)
    df1=pd.DataFrame(np1_train)
    ax=df1[0].plot(title='Bagging',label='train',color='r')
    df2[0].plot(label='test',ax=ax,color='g')
    ax.legend(loc='upper right')
    ax.set_xlabel('Bags')
    ax.set_ylabel('RMSE')
    plt.show()
'''
