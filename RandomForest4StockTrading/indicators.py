import pandas as pd
import numpy as np
import datetime as dt
from util import get_data
import matplotlib.pyplot as plt

def bollinger_bands(df_price, window=20, k=2, plot_start_date='2008-01-01', plot_end_date='2009-12-31'):
    sma = df_price.rolling(window).mean()
    std = df_price.rolling(window).std()
    upper_band = sma + k * std
    lower_band = sma - k * std
    bb = pd.concat([lower_band, upper_band], axis=1)
    bb.columns = ['lowerband', 'upperband']

    bb_range = bb.loc[plot_start_date:plot_end_date]
    df_price_range = df_price[plot_start_date:plot_end_date]
    sma_range = sma.loc[plot_start_date:plot_end_date]
    std_range = std.loc[plot_start_date:plot_end_date]

    # BBP calculation
    # bbp_range = (df_price_range - sma_range) / (2 * std_range)
    # bbp = (df_price - sma) / (2 * std)
    bbp = (df_price - lower_band) / (upper_band - lower_band)

    # # plot Bollinger Bands and BBP
    # fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(10, 8))
    # ax1.plot(df_price_range.index, df_price_range, label='Price')
    # ax1.plot(sma_range.index, sma_range, label=f'SMA({window})')
    # ax1.plot(bb_range.index, bb_range['upperband'], label=f'Upper Band({window}, {k})')
    # ax1.plot(bb_range.index, bb_range['lowerband'], label=f'Lower Band({window}, {k})')
    # ax1.set_title(f'Bollinger Bands({window}, {k})')
    # ax1.set_xlabel('Date')
    # ax1.set_ylabel('Price')
    # ax1.legend()
    #
    # ax2.set_title('BBP')
    # ax2.plot(bbp.index, bbp, label=f'BBP({window}, {k})')
    # ax2.axhline(y=1, color='g', linestyle='--')
    # ax2.axhline(y=-1, color='r', linestyle='--')
    # ax2.set_xlabel('Date')
    # ax2.set_ylabel('BBP')
    # ax2.legend()
    #
    # plt.savefig(f'BB_{window}_{k}.png')
    # plt.show()

    return bbp


def simple_moving_average(df_price, window=20, plot_start_date='2008-01-01', plot_end_date='2009-12-31'):
    sma = df_price.rolling(window).mean()
    price_sma = df_price / sma

    sma_range = sma[plot_start_date:plot_end_date]
    df_price_range = df_price[plot_start_date:plot_end_date]
    price_sma_range = df_price_range / sma_range

    # ax2.set_title(f'Price/SMA({window})')
    # ax2.plot(df_price_range.index, price_sma, label=f'Price/SMA({window})')
    # ax2.axhline(y=1, color='r', linestyle='--')
    # ax2.set_xlabel('Date')
    # ax2.set_ylabel(f'Price/SMA({window})')
    # ax2.legend()
    #
    # plt.savefig(f'SMA_{window}.png')
    # # plt.show()

    return sma, price_sma


def momentum(df_price, window=10, plot_start_date='2008-01-01', plot_end_date='2009-12-31'):
    momentum = df_price.diff(window)
    momentum_range = momentum.loc[plot_start_date:plot_end_date]
    df_price_range = df_price.loc[plot_start_date:plot_end_date]

    # # plot Momentum
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    #
    # # plot price data on the first subplot
    # ax1.plot(df_price_range.index, df_price_range, label='Price')
    # ax1.set_title('Price')
    # ax1.legend()
    # ax1.set_ylabel('Price')
    # ax1.grid()
    #
    # # plot momentum on the second subplot
    # ax2.plot(momentum_range.index, momentum_range, label=f'Momentum({window})', color='orange')
    # ax2.set_title(f'Momentum({window})')
    # ax2.set_xlabel('Date')
    # ax2.set_ylabel('Momentum')
    # ax2.axhline(y=0, color='r', linestyle='--')
    # plt.savefig(f'MM_{window}.png')
    # # plt.show()

    return momentum


def golden_cross(df_price, short_window=10, long_window=100, plot_start_date='2008-01-01', plot_end_date='2009-12-31'):
    short_sma = df_price.rolling(short_window).mean()
    long_sma = df_price.rolling(long_window).mean()

    short_sma_range = short_sma.loc[plot_start_date:plot_end_date]
    long_sma_range = long_sma.loc[plot_start_date:plot_end_date]
    df_price_range = df_price.loc[plot_start_date:plot_end_date]

    # Calculate crossover points
    crossover_points = np.sign(short_sma - long_sma).diff()
    buy_points = df_price_range[crossover_points > 0]
    sell_points = df_price_range[crossover_points < 0]

    # # plot Golden Cross
    # fig, ax = plt.subplots(figsize=(10, 5))
    # ax.plot(df_price_range.index, df_price_range, label='Price')
    # ax.plot(short_sma_range.index, short_sma_range, label=f'Short SMA({short_window})')
    # ax.plot(long_sma_range.index, long_sma_range, label=f'Long SMA({long_window})',color='purple')
    # ax.scatter(buy_points.index, buy_points, marker='^', color='g', label='Buy')
    # ax.scatter(sell_points.index, sell_points, marker='v', color='r', label='Sell')
    # ax.set_title(f'Golden Cross({short_window}, {long_window})')
    # ax.set_xlabel('Date')
    # ax.set_ylabel('Price')
    # ax.legend()
    # plt.savefig(f'GC_{short_window}_{long_window}.png')
    # # plt.show()

    return short_sma, long_sma


def macd(df_prices, short_window=12, long_window=26, signal_window=9, plot_start_date='2008-01-01', plot_end_date='2009-12-31'):
    df_price_range = df_prices[plot_start_date:plot_end_date]
    short_ema = df_prices.ewm(span=short_window, adjust=False).mean()
    long_ema = df_prices.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    histogram = macd - signal
    histogram = np.array(histogram).ravel()  # reshape to 1-dimensional array
    histogram = pd.Series(histogram, index=df_prices.index)  # convert to pandas Series
    macd_range = macd[plot_start_date:plot_end_date]
    signal_range = signal[plot_start_date:plot_end_date]
    histogram_range = histogram[plot_start_date:plot_end_date]

    # # Create two subplots with shared x-axis
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    #
    # # Plot the price data on the first subplot
    # ax1.plot(df_price_range.index, df_price_range, label='Price')
    # ax1.legend()
    # ax1.set_ylabel('Price')
    # ax1.set_title('Price')
    # ax1.grid()
    #
    # # Plot the MACD indicators on the second subplot
    # ax2.plot(macd_range.index, macd_range, label=f'MACD({short_window},{long_window},{signal_window})')
    # ax2.plot(signal_range.index, signal_range, label=f'Signal Line({signal_window})')
    #
    # # Set the color of the bars based on the sign of the histogram
    # colors = ['g' if h > 0 else 'r' for h in histogram_range]
    # ax2.bar(histogram_range.index, histogram_range, label='Histogram', width=0.7, linewidth=1, color=colors)
    # ax2.legend()
    # ax2.set_title(f'MACD({short_window},{long_window},{signal_window})')
    # ax2.set_xlabel('Date')
    # ax2.set_ylabel('MACD')
    # ax2.grid()
    #
    # # plt.show()
    # plt.savefig(f'MACD_{short_window}_{long_window}_{signal_window}.png')

    return macd, signal, histogram


def author():
    return "ycheng345"


def test_code():
    sd = dt.datetime(2007, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    syms = ['JPM']
    dates = pd.date_range(sd, ed)
    df_prices = get_data(syms, dates)
    df_prices = df_prices[syms]

    # Calculate indicators
    upper_band, lower_band, bbp = bollinger_bands(df_prices) # Done
    sma = simple_moving_average(df_prices) # Done
    mom = momentum(df_prices) # Done
    short_sma, long_sma = golden_cross(df_prices) # Done
    macd_signal, macd_ema, histogram = macd(df_prices)


if __name__ == "__main__":
    test_code()



