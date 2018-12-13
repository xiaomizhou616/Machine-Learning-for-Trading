import numpy as np

def moving_average(vals, window_size=20):
    value = np.cumsum(vals, dtype=float)
    value[window_size:] = value[window_size:] - value[:-window_size]
    return value[window_size-1:-1] / window_size

def simple_moving_average(prices, window_size=20):
    prices = prices / prices.ix[0]
    price_val = np.array(prices.values)
    moving_avg = np.concatenate((np.array([np.nan] * (window_size)), moving_average(price_val, window_size = window_size)))

    sma = price_val / moving_avg - 1
    return sma

def bollinger_band(prices, window_size = 10):
    sma_in = [0] * len(prices.index)
    prices = prices / prices.ix[0]
    price_val = np.array(prices.values)
    moving_avg = np.concatenate((np.array([np.nan] * (window_size)), moving_average(price_val, window_size = window_size)))
    moving_std = np.array([np.nan] * window_size + [price_val[start : start + window_size].std() for start in range(len(sma_in) - window_size)])

    indi_val = (price_val - moving_avg) / (moving_std * 2)
    return indi_val

def momentum(prices, window_size):
    prices = prices / prices.ix[0]
    price_val = np.array(prices.values)

    indi_val = np.array(prices.diff(window_size)/prices.shift(window_size))
    return indi_val

if __name__=="__main__":
    pass