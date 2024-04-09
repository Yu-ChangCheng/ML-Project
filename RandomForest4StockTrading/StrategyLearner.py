""""""  		  	   		  		 			  		 			     			  	 
"""  		  	   		  		 			  		 			     			  	 
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  		 			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		  		 			  		 			     			  	 
All Rights Reserved  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
Template code for CS 4646/7646  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  		 			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		  		 			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		  		 			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		  		 			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		  		 			  		 			     			  	 
or edited.  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
We do grant permission to share solutions privately with non-students such  		  	   		  		 			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		  		 			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  		 			  		 			     			  	 
GT honor code violation.  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
-----do not edit anything above this line---  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
Student Name: Yu-Chang (replace with your name)  		  	   		  		 			  		 			     			  	 
GT User ID: ycheng345 (replace with your User ID)  		  	   		  		 			  		 			     			  	 
GT ID: 903326976 (replace with your GT ID)  		  	   		  		 			  		 			     			  	 
"""  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
import datetime as dt  		  	   		  		 			  		 			     			  	 
import random  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
import pandas as pd  		  	   		  		 			  		 			     			  	 
import util as ut
from util import get_data
import RTLearner as rt
import BagLearner as bl
import numpy as np
import matplotlib.pyplot as plt
import marketsimcode
import indicators

class StrategyLearner(object):  		  	   		  		 			  		 			     			  	 
    """  		  	   		  		 			  		 			     			  	 
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		  		 			  		 			     			  	 
        If verbose = False your code should not generate ANY output.  		  	   		  		 			  		 			     			  	 
    :type verbose: bool  		  	   		  		 			  		 			     			  	 
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		  		 			  		 			     			  	 
    :type impact: float  		  	   		  		 			  		 			     			  	 
    :param commission: The commission amount charged, defaults to 0.0  		  	   		  		 			  		 			     			  	 
    :type commission: float  		  	   		  		 			  		 			     			  	 
    """  		  	   		  		 			  		 			     			  	 
    # constructor  		  	   		  		 			  		 			     			  	 
    def __init__(self, verbose=False, impact=0.0, commission=0.0):
        self.verbose = verbose  		  	   		  		 			  		 			     			  	 
        self.impact = impact  		  	   		  		 			  		 			     			  	 
        self.commission = commission
        self.leaf_size = 5
        self.bags = 100
        self.learner = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": self.leaf_size}, bags=self.bags, boost=False, verbose=False)
        self.days_for_return = 3
        self.Y_BUY = 0.025
        self.Y_SELL = -0.015

    def generate_train_data(symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000, days_for_return=10, impact=0.005, Y_BUY=0.02, Y_SELL=-0.02):

        # Get price data from 2 month before to calculate the indicator
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

        X_data = pd.concat([bbp, price_sma, mm], axis=1)
        X_data.columns = ['bbp', 'price_sma', 'mm']

        prices_lenth = len(prices)
        Y_data = np.zeros(prices_lenth - days_for_return)

        for t in range(prices_lenth - days_for_return):
            daily_return = (prices[t + days_for_return] / prices[t]) - 1.0

            if daily_return > Y_BUY + impact:
                Y_data[t] = 1  # LONG
            elif daily_return < Y_SELL - impact:
                Y_data[t] = -1  # SHORT
            else:
                Y_data[t] = 0  # CASH

        # Truncate X_data to match the length of Y_data
        X_data = X_data.iloc[:prices_lenth - days_for_return]

        return X_data, Y_data

    def generate_test_data(self, symbol='JPM', sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 1, 1), sv=100000):
        # Get price data from 2 months before to calculate the indicator
        adjusted_start_date = sd - pd.DateOffset(months=2)
        dates = pd.date_range(adjusted_start_date, ed)
        prices_all = get_data([symbol], dates, addSPY=True, colname='Adj Close')
        prices = prices_all[symbol]

        # Get indicators
        bbp = indicators.bollinger_bands(prices, window=30, k=2)
        sma, price_sma = indicators.simple_moving_average(prices, window=30)
        mm = indicators.momentum(prices, window=30)

        # Convert it back to the original date
        prices = prices[sd:ed]
        bbp = bbp[sd:ed]
        price_sma = price_sma[sd:ed]
        mm = mm[sd:ed]

        X_data = pd.concat([bbp, price_sma, mm], axis=1)
        X_data.columns = ['bbp', 'price_sma', 'mm']

        return X_data

    def add_evidence(
        self,
        symbol="IBM",
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 1, 1),
        sv=10000,
    ):
        X_data, Y_data = StrategyLearner.generate_train_data(symbol=symbol, sd=sd, ed=ed, sv=sv,
                                                             days_for_return=self.days_for_return, impact=self.impact,
                                                             Y_BUY=self.Y_BUY, Y_SELL=self.Y_SELL)

        self.learner.add_evidence(X_data.values, Y_data)

    # this method should use the existing policy and test it against new data
    def testPolicy(
        self,
        symbol="IBM",
        sd=dt.datetime(2009, 1, 1),
        ed=dt.datetime(2010, 1, 1),
        sv=10000,
    ):

        X_test = self.generate_test_data(symbol=symbol, sd=sd, ed=ed, sv=sv)

        preds = self.learner.query(X_test.values)

        holdings = pd.DataFrame(index=X_test.index, columns=['Shares'])
        holdings['Shares'] = 0

        holdings.loc[preds == 1, 'Shares'] = 1000
        holdings.loc[preds == -1, 'Shares'] = -1000

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
    strategy_learner = StrategyLearner(verbose = False, impact = 0.005, commission=9.95)
    # In-sample training and testing
    strategy_learner.add_evidence(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    trades_is = strategy_learner.testPolicy(symbol = 'JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    # print(trades_is)
    sl_port_vals_is = marketsimcode.compute_portvals(orders=trades_is, start_val=100000, commission=9.95, impact=0.005)
    # print(sl_port_vals_is)
    sl_port_vals_normed_is = sl_port_vals_is / sl_port_vals_is.iloc[0]

    # In-sample Benchmark
    benchmark_trades = pd.DataFrame(index=sl_port_vals_is.index, columns=['Shares'])
    benchmark_trades['Shares'] = 0
    benchmark_trades.iloc[0] = 1000
    benchmark_trades.iloc[-1] = -1000
    # print(benchmark_trades)
    benchmark_portval_normed_is = marketsimcode.compute_portvals(orders=benchmark_trades, commission=9.95, impact=0.005, start_val=100000)
    # print(benchmark_portval_normed_is)
    benchmark_portval_normed_is = benchmark_portval_normed_is / benchmark_portval_normed_is.iloc[0]

    # Plotting in-sample data
    plt.figure(figsize=(12, 8))
    plt.plot(sl_port_vals_normed_is.index, sl_port_vals_normed_is, label='Strategy Learner', color='red')
    plt.plot(benchmark_portval_normed_is.index, benchmark_portval_normed_is, label='Benchmark', color='purple')

    long_entries = trades_is[trades_is['Shares'] > 0].index
    short_entries = trades_is[trades_is['Shares'] < 0].index

    for entry in long_entries:
        plt.axvline(entry, color='blue', lw=1)

    for entry in short_entries:
        plt.axvline(entry, color='black', lw=1)

    plt.xlabel('Date')
    plt.ylabel('Normalized Portfolio Value')
    plt.title('Strategy Learner vs. Benchmark (In-Sample)')
    plt.legend()
    plt.grid()
    plt.savefig('images/SL_vs_Benchmark_in-sample.png')
    plt.clf()


    """
    out-sample
    """
    # Out-of-sample testing
    trades_os = strategy_learner.testPolicy(symbol='JPM', sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31),
                                            sv=100000)
    sl_port_vals_os = marketsimcode.compute_portvals(orders=trades_os, start_val=100000, commission=9.95, impact=0.005)
    sl_port_vals_normed_os = sl_port_vals_os / sl_port_vals_os.iloc[0]

    # Out-of-Sample Benchmark
    benchmark_trades_os = pd.DataFrame(index=sl_port_vals_os.index, columns=['Shares'])
    benchmark_trades_os['Shares'] = 0
    benchmark_trades_os.iloc[0] = 1000
    benchmark_trades_os.iloc[-1] = -1000
    benchmark_portval_os = marketsimcode.compute_portvals(orders=benchmark_trades_os, commission=9.95, impact=0.005, start_val=100000)
    benchmark_portval_normed_os = benchmark_portval_os / benchmark_portval_os.iloc[0]

    # Out-of-Sample Plot
    plt.figure(figsize=(12, 8))
    plt.plot(sl_port_vals_normed_os.index, sl_port_vals_normed_os, label='Strategy Learner', color='red')
    plt.plot(benchmark_portval_normed_os.index, benchmark_portval_normed_os, label='Benchmark', color='purple')

    long_entries_os = trades_os[trades_os['Shares'] > 0].index
    short_entries_os = trades_os[trades_os['Shares'] < 0].index

    for entry in long_entries_os:
        plt.axvline(entry, color='blue', lw=1)

    for entry in short_entries_os:
        plt.axvline(entry, color='black', lw=1)

    plt.xlabel('Date')
    plt.ylabel('Normalized Portfolio Value')
    plt.title('Strategy Learner vs. Benchmark (Out-of-Sample)')
    plt.legend()
    plt.grid()
    plt.savefig('images/SL_vs_Benchmark_out-of-sample.png')
    plt.clf()

    sl_is_cum_ret, sl_is_avg_daily_ret, sl_is_std_daily_ret, sl_is_sharpe_ratio = compute_portfolio_stats(sl_port_vals_normed_is)
    b_is_cum_ret, b_is_avg_daily_ret, b_is_std_daily_ret, b_is_sharpe_ratio = compute_portfolio_stats(benchmark_portval_normed_is)
    sl_os_cum_ret, sl_os_avg_daily_ret, sl_os_std_daily_ret, sl_os_sharpe_ratio = compute_portfolio_stats(sl_port_vals_normed_os)
    b_os_cum_ret, b_os_avg_daily_ret, b_os_std_daily_ret, b_os_sharpe_ratio = compute_portfolio_stats(benchmark_portval_normed_os)

    columns = pd.MultiIndex.from_product([['In-Sample', 'Out-of-Sample'], ['Strategy Learner', 'Benchmark']],
                                         names=['Period', 'Strategy'])

    stats = pd.DataFrame(columns=columns,
                         index=['Cumulative Return', 'Average Daily Return', 'Std Daily Return', 'Sharpe Ratio'])

    stats.loc['Cumulative Return'] = [sl_is_cum_ret, b_is_cum_ret, sl_os_cum_ret, b_os_cum_ret]
    stats.loc['Average Daily Return'] = [sl_is_avg_daily_ret, b_is_avg_daily_ret, sl_os_avg_daily_ret, b_os_avg_daily_ret]
    stats.loc['Std Daily Return'] = [sl_is_std_daily_ret, b_is_std_daily_ret, sl_os_std_daily_ret, b_os_std_daily_ret]
    stats.loc['Sharpe Ratio'] = [sl_is_sharpe_ratio, b_is_sharpe_ratio, sl_os_sharpe_ratio, b_os_sharpe_ratio]

    stats.to_csv('results/sl_vs_benchmark_portfolio_stats.csv')

if __name__ == "__main__":  		  	   		  		 			  		 			     			  	 
    print("One does not simply think up a strategy")
    testcode()
