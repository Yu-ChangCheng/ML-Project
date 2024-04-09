from util import get_data
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import marketsimcode
import indicators

class ManualStrategy(object):

    def testPolicy(self, symbol = 'JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000):
        # Get price data from 1 month before to calculate the indicator
        adjusted_start_date = sd - pd.DateOffset(months=2)
        dates = pd.date_range(adjusted_start_date, ed)
        prices_all = get_data([symbol], dates, addSPY=True, colname='Adj Close')
        prices = prices_all[symbol]

        # Get indicators
        bbp = indicators.bollinger_bands(prices, window=30, k=2)
        sma, price_sma = indicators.simple_moving_average(prices, window=30)
        mm = indicators.momentum(prices, window=30)

        # Convert it back to original date
        prices = prices[sd:ed]
        bbp = bbp[sd:ed]
        price_sma = price_sma[sd:ed]
        mm = mm[sd:ed]

        holdings = pd.DataFrame(index=prices.index, columns=['Shares'])
        holdings['Shares'] = np.nan

        # print(bbp)
        # print("BBP",min(bbp))
        # print("BBP", max(bbp))
        # print("Price_sma",min(price_sma))
        # print("Price_sma",max(price_sma))
        # print("mm",min(mm))
        # print("mm",max(mm))

        for i in range(prices.shape[0]):
            date = prices.index[i]
            if (bbp[date] > 0.8 or price_sma[date] > 1.2 or mm[date] > 5):
                holdings.loc[date, 'Shares'] = -1000
            elif (bbp[date] < 0.2 or price_sma[date] < 0.8 or mm[date] < -5):
                holdings.loc[date, 'Shares'] = 1000

        # turn holding to trades
        # fill the position if no trades occurs
        holdings.ffill(inplace=True)
        # fill the nan value if any missing value at beginning
        holdings.fillna(0, inplace=True)
        trades = holdings.diff().fillna(holdings.iloc[0])
        trades.columns = ['Shares']
        return trades

    def author(self):
        return "ycheng345"

def compute_portfolio_stats(port_val, rfr=0.0, sf=252.0):
    daily_returns = (port_val / port_val.shift(1)) - 1
    daily_returns = daily_returns[1:]

    cum_ret = (port_val[-1] / port_val[0]) - 1

    # Sharpe ratio
    avg_daily_ret = daily_returns.mean()
    std_daily_ret = daily_returns.std()
    k = np.sqrt(sf)
    sharpe_ratio = k * (avg_daily_ret - rfr) / std_daily_ret

    return cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio

def testcode():
    manual_strategy = ManualStrategy()
    trades_is = manual_strategy.testPolicy(symbol = 'JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    # print(trades_is)
    ms_port_vals_is = marketsimcode.compute_portvals(orders=trades_is, start_val=100000, commission=9.95, impact=0.005)
    # print(ms_port_vals_is)
    ms_port_vals_normed_is = ms_port_vals_is / ms_port_vals_is.iloc[0]

    benchmark_trades = pd.DataFrame(index=ms_port_vals_is.index, columns=['Shares'])
    benchmark_trades['Shares'] = 0
    benchmark_trades.iloc[0] = 1000
    benchmark_trades.iloc[-1] = -1000
    # print(benchmark_trades)
    benchmark_portval_normed_is = marketsimcode.compute_portvals(orders=benchmark_trades, commission=9.95, impact=0.005, start_val=100000)
    # print(benchmark_portval_normed_is)
    benchmark_portval_normed_is = benchmark_portval_normed_is / benchmark_portval_normed_is.iloc[0]


    plt.figure(figsize=(12, 8))
    plt.plot(ms_port_vals_normed_is.index, ms_port_vals_normed_is, label='Manual Strategy', color='red')
    plt.plot(benchmark_portval_normed_is.index, benchmark_portval_normed_is, label='Benchmark', color='purple')

    long_entries = trades_is[trades_is['Shares'] > 0].index
    short_entries = trades_is[trades_is['Shares'] < 0].index

    for entry in long_entries:
        plt.axvline(entry, color='blue', lw=1)

    for entry in short_entries:
        plt.axvline(entry, color='black', lw=1)

    plt.xlabel('Date')
    plt.ylabel('Normalized Portfolio Value (JPM)')
    plt.title('Manual Strategy vs. Benchmark (In-Sample)')
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig('images/MS_vs_Benchmark_in-sample.png')
    plt.clf()

    '''
    Out Sample
    '''
    # Out-of-Sample
    trades_os = manual_strategy.testPolicy(symbol='JPM', sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31),
                                           sv=100000)
    ms_port_vals_os = marketsimcode.compute_portvals(orders=trades_os, start_val=100000, commission=9.95, impact=0.005)
    ms_port_vals_normed_os = ms_port_vals_os / ms_port_vals_os.iloc[0]


    # Out-of-Sample Benchmark
    benchmark_trades_os = pd.DataFrame(index=ms_port_vals_os.index, columns=['Shares'])
    benchmark_trades_os['Shares'] = 0
    benchmark_trades_os.iloc[0] = 1000
    benchmark_trades_os.iloc[-1] = -1000
    benchmark_portval_os = marketsimcode.compute_portvals(orders=benchmark_trades_os, commission=9.95, impact=0.005,
                                                          start_val=100000)
    benchmark_portval_normed_os = benchmark_portval_os / benchmark_portval_os.iloc[0]



    # Out-of-Sample Plot
    plt.figure(figsize=(12, 8))
    plt.plot(ms_port_vals_normed_os.index, ms_port_vals_normed_os, label='Manual Strategy', color='red')
    plt.plot(benchmark_portval_normed_os.index, benchmark_portval_normed_os, label='Benchmark', color='purple')

    long_entries_os = trades_os[trades_os['Shares'] > 0].index
    short_entries_os = trades_os[trades_os['Shares'] < 0].index

    for entry in long_entries_os:
        plt.axvline(entry, color='blue', lw=1)

    for entry in short_entries_os:
        plt.axvline(entry, color='black', lw=1)

    plt.xlabel('Date')
    plt.ylabel('Normalized Portfolio Value (JPM)')
    plt.title('Manual Strategy vs. Benchmark (Out-of-Sample)')
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig('images/MS_vs_Benchmark_out-of-sample.png')
    plt.clf()


    ms_is_cum_ret, ms_is_avg_daily_ret, ms_is_std_daily_ret, ms_is_sharpe_ratio = compute_portfolio_stats(
        ms_port_vals_normed_is)
    b_is_cum_ret, b_is_avg_daily_ret, b_is_std_daily_ret, b_is_sharpe_ratio = compute_portfolio_stats(
        benchmark_portval_normed_is)
    ms_os_cum_ret, ms_os_avg_daily_ret, ms_os_std_daily_ret, ms_os_sharpe_ratio = compute_portfolio_stats(
        ms_port_vals_normed_os)
    b_os_cum_ret, b_os_avg_daily_ret, b_os_std_daily_ret, b_os_sharpe_ratio = compute_portfolio_stats(
        benchmark_portval_normed_os)
    columns = pd.MultiIndex.from_product([['In-Sample', 'Out-of-Sample'], ['Manual', 'Benchmark']],
                                         names=['Period', 'Strategy'])

    stats = pd.DataFrame(columns=columns,
                         index=['Cumulative Return', 'Average Daily Return', 'Std Daily Return', 'Sharpe Ratio'])

    stats.loc['Cumulative Return'] = [ms_is_cum_ret, b_is_cum_ret, ms_os_cum_ret, b_os_cum_ret]
    stats.loc['Average Daily Return'] = [ms_is_avg_daily_ret, b_is_avg_daily_ret, ms_os_avg_daily_ret,
                                         b_os_avg_daily_ret]
    stats.loc['Std Daily Return'] = [ms_is_std_daily_ret, b_is_std_daily_ret, ms_os_std_daily_ret, b_os_std_daily_ret]
    stats.loc['Sharpe Ratio'] = [ms_is_sharpe_ratio, b_is_sharpe_ratio, ms_os_sharpe_ratio, b_os_sharpe_ratio]

    # stats.to_csv('results/ml_vs_benchmark_portfolio_stats.csv')
    with open('p8_results.txt', 'a') as f:
        f.write("\nManual Strategy Stats:\n")
        f.write(stats.to_string())



if __name__ == "__main__":
    testcode()
