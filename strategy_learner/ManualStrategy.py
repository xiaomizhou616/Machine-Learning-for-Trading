import numpy as np
import pandas as pd
import datetime as dt
import math
import types
import os
from marketsimcode import marketsim
from util import get_data, plot_data
import indicators as indi

def print_info(portvals):
    portvals = portvals / portvals.ix[0]
    daily_returns = portvals / portvals.shift(1) - 1
    daily_returns = daily_returns[1:]

    print 'cumulative return: ' + str(float(portvals.values[-1] / portvals.values[0]) - 1)
    print 'Stdev of daily returns: ' + str(float(daily_returns.std()))
    print 'Mean of daily returns: ' + str(float(daily_returns.mean()))

def testPolicy(symbol = 'JPM', sd = dt.datetime(2008, 1, 1), ed = dt.datetime(2009, 12, 31), sv = 100000):
    prices = get_data([symbol], pd.date_range(sd, ed))
    prices = prices[symbol]
    df_trades = pd.DataFrame(data=np.zeros(len(prices.index)), index=prices.index, columns = ['val'])

    current = 0
    window_size = 20

    sma = indi.simple_moving_average(prices, window_size=window_size)
    for i in range(window_size, len(prices.index)):
        # the threshold here is 0.1
        # Smaller than threshold -> buy
        # Bigger than threshold -> sell
        if sma[i] < -0.1:
            df_trades['val'].iloc[i] = 1000 - current
            current = 1000
        elif sma[i] > 0.1:
            df_trades['val'].iloc[i] = - current - 1000
            current = -1000

    return df_trades

if __name__ == '__main__':
    symbols = ['JPM']
    start_date = '2008-01-01'
    end_date = '2009-12-31'
#     #for out-sample time using following sd and ed
    # start_date = '2010-01-01'
#     #end_date = '2011-12-31'
    prices = get_data(symbols, pd.date_range(start_date, end_date))
    prices = prices[symbols]
#     # print prices
    df_trades = testPolicy(sd=start_date, ed=end_date)
#     df_joined = df_trades.join(prices, lsuffix='_best', rsuffix='_benchmark')

    portvals = marketsim(symbols[0], df_trades, commission = 9.95, impact = 0.005)

    print_info(portvals)
#     # portvals = marketsim(df_trades, prices, commission = 0, impact = 0)
#     df_joined = df_joined.join(portvals, lsuffix='_best', rsuffix = 'whatever')
#     prices_val = prices.values

#     # to generate Benchmark values
#     d = np.zeros(len(prices.index))
#     d[0] = 1000
#     df_trade_none = pd.DataFrame(data=d, index=prices.index, columns = ['val'])
#     port_benchmark = marketsim(df_trade_none, prices)

#     portvals = portvals / portvals.ix[0]
#     port_benchmark = port_benchmark / port_benchmark.ix[0]

#     print "My strategy performance"
#     print_info(portvals)
#     print "Benchmark performance"
#     print_info(port_benchmark)

#     benchmark, = plt.plot(port_benchmark, 'b')
#     mystrategy, = plt.plot(portvals, 'k')
#     for i in range(len(prices.index)):
#         if df_trades['val'].iloc[i] > 0:
#             plt.axvline(x=prices.index[i], c = 'g')
#         elif df_trades['val'].iloc[i] < 0:
#             plt.axvline(x=prices.index[i], c = 'r')
# #generate the graphs
#     plt.gcf().subplots_adjust(bottom= 0.2)
#     plt.legend([benchmark, mystrategy], ['Benchmark', 'My Strategy'],loc='upper right')
#     plt.ylabel('Value')
#     plt.xlabel('Date')
#     plt.xticks(rotation = 45)
#     plt.title("My Strategy VS Benchmark (Development Period)")
#     plt.show()