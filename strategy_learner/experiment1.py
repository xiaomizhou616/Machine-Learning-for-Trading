"""
Experiment1
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

if __name__== "__main__":
    sd = dt.datetime(2008,1,1)
    ed = dt.datetime(2009,12,31)
    dates = pd.date_range(sd,ed)
    symbol = 'JPM'
    prices_all = ut.get_data([symbol],dates)
    prices = prices_all[symbol]
    benchmark_record = prices_all[[symbol,]].copy()
    benchmark_record.values[:, :] = 0
    benchmark_record.values[0, :] = 1000
    values_benchmark = marketsim(symbol, benchmark_record)
    rtl = sl.StrategyLearner(verbose = False, impact = 0)
    rtl.addEvidence(symbol = symbol, sd = sd, ed = ed, sv = 100000)
    trades_rtl = rtl.testPolicy(symbol = symbol,sd = sd, ed = ed, sv = 100000)
    values_rtl = marketsim(symbol, trades_rtl)
    trades_manual = manu.testPolicy(symbol = symbol, sd = sd, ed = ed)
    values_manual = marketsim(symbol, trades_manual)
    values_benchmark = values_benchmark / values_benchmark.iloc[0]
    values_manual = values_manual / values_manual.iloc[0]
    values_rtl = values_rtl / values_rtl.iloc[0]
    benchmark_record, = plt.plot(values_benchmark, "b", label = "benchmark")
    rtl, = plt.plot(values_rtl, "g", label = "RTLearner")
    manual_strategy, = plt.plot(values_manual, "r", label = "Manual Strategy")
    plt.legend(handles=[benchmark_record, manual_strategy, rtl], loc=0)
    plt.gcf().subplots_adjust(left=0.1, bottom=0.05, right=0.125, top=0.1, wspace=0.2, hspace=0.2)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title("RTLearner VS. Manual Strategy")
    plt.xticks(rotation="vertical")
    plt.show()
