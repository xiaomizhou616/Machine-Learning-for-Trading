import pandas as pd
import numpy as np
import datetime as dt
import os
import StrategyLearner as st

from util import get_data, plot_data
 
def marketsim(symbol, trades_df, start_val = 1000000, commission=9.95, impact=0.005):
    start_date = trades_df.index[0]
    end_date = trades_df.index[-1]

    dates = pd.date_range(start_date, end_date)

    prices = get_data([symbol],dates)  
    prices.fillna(method = 'ffill',inplace = True)
    prices.fillna(method = 'bfill',inplace = True)
    prices = prices[[symbol]]

    prices['cash'] = 1.0
    trader = prices.copy()
    trader[:] = 0.0

    for order_date, row in trades_df.iterrows():
        order_symbol = symbol
        order_share = row.iloc[0]
 
        trader.loc[order_date,order_symbol] =trader.loc[order_date,order_symbol]+order_share
        share_price = prices.at[order_date,order_symbol]
        trader.loc[order_date,'cash'] = trader.loc[order_date,'cash'] - order_share * share_price
        trader.loc[order_date,'cash'] = trader.loc[order_date,'cash'] - (commission + abs(order_share) * share_price * impact)
 
    temporary = trader.copy()
    temporary[:] = 0.0
    temporary.loc[start_date].at['cash'] = start_val
    temporary.iloc[0,:] = temporary.iloc[0,:]+trader.iloc[0,:]
    number=len(temporary.index)
    for i in range(1,number):
        temporary.iloc[i,:] = temporary.iloc[i,:]+(temporary.iloc[i-1,:] + trader.iloc[i,:])
 
    stock_value = pd.DataFrame(temporary.values * prices.values , index=prices.index, columns= prices.columns)
    portvals = stock_value.sum(axis=1)
 
    return portvals

def test_code():
    symbol='JPM'

    # Test strategy_learner
    learner = st.StrategyLearner()
    insample_args=dict(symbol=symbol, sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv=100000)
    learner.addEvidence(**insample_args)
    trades_df = learner.testPolicy(**insample_args)

    portvals = marketsim(symbol, trades_df, start_val=insample_args['sv'])
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"
 
    print "Final Portfolio Value: {}".format(portvals[-1])

    portvals = portvals / portvals.ix[0]
    daily_returns = portvals / portvals.shift(1) - 1
    daily_returns = daily_returns[1:]
    print 'cumulative return: ' + str(float(portvals.values[-1] / portvals.values[0]) - 1)
    print 'Stdev of daily returns: ' + str(float(daily_returns.std()))
    print 'Mean of daily returns: ' + str(float(daily_returns.mean()))

    # Test manual_strategy

 
def author():
    return "xhan306"
 
if __name__ == "__main__":
    test_code()