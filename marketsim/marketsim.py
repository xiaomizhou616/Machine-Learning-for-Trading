"""MC2-P1: Market simulator.

Copyright 2017, Georgia Tech Research Corporation
Atlanta, Georgia 30332-0415
All Rights Reservend_date
"""

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data

def compute_portvals(orders_file = "./orders/orders.csv", start_val = 1000000, commission=9.95, impact=0.005):
    # this is the function the autograder will call to test your code
    # NOTE: orders_file may be a string, or it may be a file object. Your
    # code should work correctly with either input
    # TODO: Your code here

    orders_record = pd.read_csv(orders_file,index_col="Date",parse_dates = True,na_values = ['nan'])
    orders_record.sort_index(inplace=True)
    start_date = orders_record.index.unique()[0]
    end_date = orders_record.index.unique()[-1]
    dates = pd.date_range(start_date,end_date)
    symbol = list(orders_record['Symbol'].unique())
    prices = get_data(symbol,dates)  
    prices.fillna(method = 'ffill',inplace = True)
    prices.fillna(method = 'bfill',inplace = True)
    prices = prices[symbol]
    prices['cash'] = 1.0
    trader = prices.copy()
    trader[:] = 0.0




    for order_date, row in orders_record.iterrows():
        order_symbol = row['Symbol']
        order_share = row['Shares']

        if row['Order']== 'SELL':
            order_share = (-1) * order_share
        else:
            order_share = order_share

        trader.loc[order_date,order_symbol] =trader.loc[order_date,order_symbol]+order_share
        share_price = prices.at[order_date,order_symbol]
        trader.loc[order_date,'cash'] = trader.loc[order_date,'cash'] -order_share * share_price
        trader.loc[order_date,'cash'] = trader.loc[order_date,'cash']-(commission + abs(order_share) * share_price * impact)

    temporary = trader.copy()
    temporary[:] = 0.0
    temporary.loc[start_date].at['cash'] = start_val
    temporary.iloc[0,:] = temporary.iloc[0,:]+trader.iloc[0,:]
    number=len(temporary.index)
    for i in range(1,number):
        temporary.iloc[i,:] = temporary.iloc[i,:]+(temporary.iloc[i-1,:] + trader.iloc[i,:])

    stock_value = pd.DataFrame(temporary.values * prices.values , index=prices.index, columns= prices.columns)
    portvals = stock_value.sum(axis=1)


    # In the template, instead of computing the value of the portfolio, we just
    # read in the value of IBM over 6 months
    '''start_date = dt.datetime(2008,1,1)
    end_date = dt.datetime(2008,6,1)
    portvals = get_data(['IBM'], pd.date_range(start_date, end_date))
    portvals = portvals[['IBM']]  # remove SPY
    rv = pd.DataFrame(index=portvals.index, data=portvals.as_matrix())
    '''

    #return rv
    return portvals

def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be callend_date.
    # Define input parameters

    of = "./orders/orders2.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file = of, start_val = sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    start_date = dt.datetime(2008,1,1)
    end_date = dt.datetime(2008,6,1)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [0.2,0.01,0.02,1.5]
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [0.2,0.01,0.02,1.5]

    # Compare portfolio against $SPX
    print "Date Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of SPY : {}".format(cum_ret_SPY)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])

def author():
    return "xhan306"

if __name__ == "__main__":
    test_code()
