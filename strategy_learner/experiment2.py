"""
Experiment2
Name: Xi Han
GTid: xhan306
"""

import datetime as dt
import pandas as pd
import numpy as np
import util as ut
import random
import BagLearner as bl
import RTLearner as rt
import indicators as idt
import ManualStrategy as manu
from marketsimcode import marketsim
import StrategyLearner as sl
import matplotlib.pyplot as plt


def information(portvals):
    daily_returns = portvals / portvals.shift(1) - 1
    daily_returns = daily_returns[1:]
    print 'cumulative return: ' + str(float(portvals.values[-1] / portvals.values[0]) - 1)
    print 'Mean of daily returns: ' + str(float(daily_returns.mean()))
    print 'Standard deviation of daily returns: ' + str(float(daily_returns.std()))
if __name__ == '__main__':
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    dates = pd.date_range(sd, ed)
    symbol = 'JPM'
    prices_all = ut.get_data([symbol], dates)
    prices = prices_all[symbol]
    summary1 = []
    trades_number = []
    impacts = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    for impact in impacts:
        rtl = sl.StrategyLearner(verbose=False, impact=impact)
        rtl.addEvidence(symbol=symbol, sd=sd, ed=ed, sv=100000)
        trades_rtl = rtl.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=100000)
        trades_number.append(np.count_nonzero(trades_rtl))
        values_rtl = marketsim(symbol, trades_rtl, impact = impact)
        values_rtl = values_rtl / values_rtl.iloc[0]
        information(values_rtl)
        print trades_number
        summary1.append(values_rtl)
    plt.xlabel('Impact')
    plt.ylabel('Number of trades')
    plt.xticks(np.arange(7), impacts)
    plt.plot(trades_number)
    plt.title("Impact VS. Number of trades")
    plt.show()
    summary2 = []
    for i in range(len(impacts)):
        rtlstrategy, = plt.plot(summary1[i], label = str(impacts[i]))
        summary2.append(rtlstrategy)
    plt.gcf().subplots_adjust(left=0.1, bottom=0.05, right=0.125, top=0.1, wspace=0.2, hspace=0.2)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title("RTLearner with different impacts")
    plt.xticks(rotation="vertical")
    plt.legend(handles=summary2, loc=3)
    plt.show()
