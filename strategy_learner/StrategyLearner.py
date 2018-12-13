"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch
Xi Han
xhan306
"""

import datetime as dt
import pandas as pd
import numpy as np
import util as ut
import random
import BagLearner as bl
import RTLearner as rt
import indicators as idt
import math
import indicators as indi

class StrategyLearner(object):

    def __init__(self, verbose = False, impact=0.0):
        self.verbose = verbose
        self.impact = impact
        self.learner = bl.BagLearner(learner = rt.RTLearner, kwargs = {"leaf_size":5}, bags = 20)
        self.window_size = 20
        self.idt_size = 5
        self.N = 10

    def addEvidence(self, symbol = "IBM", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,1,1), \
        sv = 10000): 

        window_size = self.window_size
        num_feature = self.idt_size
        num_n_day_return = 10
        threshold = max(0.03, 2 *self.impact)

        prices = ut.get_data([symbol], pd.date_range(sd, ed))
        prices = prices[symbol]

        sma = indi.simple_moving_average(prices, window_size=window_size)

        X = []
        Y = []
        for i in range(window_size + num_feature + 1, len(prices) - num_n_day_return):
            X.append(np.array(sma[i - num_feature:i]))

            gain = (prices.values[i + num_n_day_return] - prices.values[i])/prices.values[i]
            if gain > threshold:
                Y.append(1)
            elif gain < -threshold:
                Y.append(-1)
            else:
                Y.append(0)
        X = np.array(X)
        Y = np.array(Y)
  
        self.learner.addEvidence(X,Y)

    def testPolicy(self, symbol = "IBM", sd=dt.datetime(2009,1,1), ed=dt.datetime(2010,1,1), sv = 10000):

        current_holding = 0
        dates = pd.date_range(sd,ed)
        prices_all = ut.get_data([symbol],dates)
        trades = prices_all[[symbol,]].copy(deep=True)

        window_size = self.window_size
        num_feature = self.idt_size
        
        prices = ut.get_data([symbol], pd.date_range(sd, ed))
        prices = prices[symbol]

        # Indicator 1: simple moving average
        indi_val = indi.simple_moving_average(prices, window_size=window_size)

        # Indicator 2: bollinger_band
        #indi_val = indi.bollinger_band(prices, window_size=window_size)

        # Indicator 3: momentum
        #indi_val = indi.momentum(prices, window_size=window_size)

        trades.values[:,:] = 0
        Xtest = []

        for i in range(window_size + num_feature +1, len(prices)-1):
            data = np.array(indi_val[i - num_feature:i])
            Xtest.append(data)

        result = self.learner.query(Xtest)
        for i,r in enumerate(result):
            if r > 0:
                trades.values[i + window_size + num_feature +1,:] = 1000- current_holding
                current_holding = 1000
            elif r < 0:
                trades.values[ i + window_size + num_feature +1,:] = -1000 - current_holding
                current_holding = -1000

        return trades

    def author():
        return "xhan306"

if __name__=="__main__":
    print "One does not simply think up a strategy"
