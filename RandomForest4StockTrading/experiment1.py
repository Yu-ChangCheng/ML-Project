import ManualStrategy
import StrategyLearner
import marketsimcode
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import os
import sys
import random


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

def testcode(seed=903326976):
    np.random.seed(seed)
    random.seed(seed)
    # manual learner in sample
    manual_strategy = ManualStrategy.ManualStrategy()
    trades_is = manual_strategy.testPolicy(symbol = 'JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    ms_port_vals_is = marketsimcode.compute_portvals(orders=trades_is, start_val=100000, commission=9.95, impact=0.005)
    ms_port_vals_normed_is = ms_port_vals_is / ms_port_vals_is.iloc[0]

    # strategy learner in sample
    strategy_learner = StrategyLearner.StrategyLearner(verbose=False, impact=0.005, commission=9.95)
    strategy_learner.add_evidence(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    trades_is = strategy_learner.testPolicy(symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31),
                                            sv=100000)
    sl_port_vals_is = marketsimcode.compute_portvals(orders=trades_is, start_val=100000, commission=9.95, impact=0.005)
    sl_port_vals_normed_is = sl_port_vals_is / sl_port_vals_is.iloc[0]

    # In-sample Benchmark
    benchmark_trades = pd.DataFrame(index=sl_port_vals_is.index, columns=['Shares'])
    benchmark_trades['Shares'] = 0
    benchmark_trades.iloc[0] = 1000
    benchmark_trades.iloc[-1] = -1000
    benchmark_portval_normed_is = marketsimcode.compute_portvals(orders=benchmark_trades, commission=9.95, impact=0.005,
                                                                 start_val=100000)
    benchmark_portval_normed_is = benchmark_portval_normed_is / benchmark_portval_normed_is.iloc[0]

    plt.figure(figsize=(12, 8))
    plt.plot(ms_port_vals_normed_is.index, ms_port_vals_normed_is, label='Manual Strategy', color='red')
    plt.plot(sl_port_vals_normed_is.index, sl_port_vals_normed_is, label='Strategy Learner', color='green')
    plt.plot(benchmark_portval_normed_is.index, benchmark_portval_normed_is, label='Benchmark', color='purple')

    plt.xlabel('Date')
    plt.ylabel('Normalized Portfolio Value')
    plt.title('Manual Learner vs. Strategy Learner vs. Benchmark (In-Sample)')
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig('images/ML_vs_SL_vs_Benchmark_in-sample.png')
    plt.clf()

    # manual learner out of sample
    trades_oos = manual_strategy.testPolicy(symbol='JPM', sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000)
    ms_port_vals_oos = marketsimcode.compute_portvals(orders=trades_oos, start_val=100000, commission=9.95, impact=0.005)
    ms_port_vals_normed_oos = ms_port_vals_oos / ms_port_vals_oos.iloc[0]

    # Out-of-sample testing
    trades_os = strategy_learner.testPolicy(symbol='JPM', sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31),
                                            sv=100000)
    sl_port_vals_os = marketsimcode.compute_portvals(orders=trades_os, start_val=100000, commission=9.95, impact=0.005)
    sl_port_vals_normed_os = sl_port_vals_os / sl_port_vals_os.iloc[0]

    # Out-of-sample Benchmark
    benchmark_trades_oos = pd.DataFrame(index=sl_port_vals_os.index, columns=['Shares'])
    benchmark_trades_oos['Shares'] = 0
    benchmark_trades_oos.iloc[0] = 1000
    benchmark_trades_oos.iloc[-1] = -1000
    benchmark_portval_normed_oos = marketsimcode.compute_portvals(orders=benchmark_trades_oos, commission=9.95, impact=0.005,
                                                                  start_val=100000)
    benchmark_portval_normed_oos = benchmark_portval_normed_oos / benchmark_portval_normed_oos.iloc[0]

    plt.figure(figsize=(12, 8))
    plt.plot(ms_port_vals_normed_oos.index, ms_port_vals_normed_oos, label='Manual Strategy', color='red')
    plt.plot(sl_port_vals_normed_os.index, sl_port_vals_normed_os, label='Strategy Learner', color='green')
    plt.plot(benchmark_portval_normed_oos.index, benchmark_portval_normed_oos, label='Benchmark', color='purple')

    plt.xlabel('Date')
    plt.ylabel('Normalized Portfolio Value')
    plt.title('Manual Learner vs. Strategy Learner vs. Benchmark (Out-of-Sample)')
    plt.legend(loc='lower left')
    plt.grid()
    plt.savefig('images/ML_vs_SL_vs_Benchmark_out-of-sample.png')
    plt.clf()

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
    plt.legend(loc='lower right')
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
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig('images/SL_vs_Benchmark_out-of-sample.png')
    plt.clf()

    # Compute portfolio statistics
    ms_is_cum_ret, ms_is_avg_daily_ret, ms_is_std_daily_ret, ms_is_sharpe_ratio = compute_portfolio_stats(
        ms_port_vals_normed_is)
    sl_is_cum_ret, sl_is_avg_daily_ret, sl_is_std_daily_ret, sl_is_sharpe_ratio = compute_portfolio_stats(
        sl_port_vals_normed_is)
    b_is_cum_ret, b_is_avg_daily_ret, b_is_std_daily_ret, b_is_sharpe_ratio = compute_portfolio_stats(
        benchmark_portval_normed_is)

    ms_os_cum_ret, ms_os_avg_daily_ret, ms_os_std_daily_ret, ms_os_sharpe_ratio = compute_portfolio_stats(
        ms_port_vals_normed_oos)
    sl_os_cum_ret, sl_os_avg_daily_ret, sl_os_std_daily_ret, sl_os_sharpe_ratio = compute_portfolio_stats(
        sl_port_vals_normed_os)
    b_os_cum_ret, b_os_avg_daily_ret, b_os_std_daily_ret, b_os_sharpe_ratio = compute_portfolio_stats(
        benchmark_portval_normed_oos)

    columns = pd.MultiIndex.from_product([['In-Sample', 'Out-of-Sample'], ['Manual', 'Strategy Learner', 'Benchmark']],
                                         names=['Period', 'Strategy'])

    stats = pd.DataFrame(columns=columns,
                         index=['Cumulative Return', 'Average Daily Return', 'Std Daily Return', 'Sharpe Ratio',
                                'Final Portfolio Value'])

    stats.loc['Cumulative Return'] = [ms_is_cum_ret, sl_is_cum_ret, b_is_cum_ret, ms_os_cum_ret, sl_os_cum_ret,
                                      b_os_cum_ret]
    stats.loc['Average Daily Return'] = [ms_is_avg_daily_ret, sl_is_avg_daily_ret, b_is_avg_daily_ret,
                                         ms_os_avg_daily_ret, sl_os_avg_daily_ret, b_os_avg_daily_ret]
    stats.loc['Std Daily Return'] = [ms_is_std_daily_ret, sl_is_std_daily_ret, b_is_std_daily_ret, ms_os_std_daily_ret,
                                     sl_os_std_daily_ret, b_os_std_daily_ret]
    stats.loc['Sharpe Ratio'] = [ms_is_sharpe_ratio, sl_is_sharpe_ratio, b_is_sharpe_ratio, ms_os_sharpe_ratio,
                                 sl_os_sharpe_ratio, b_os_sharpe_ratio]

    # Add final portfolio value to stats DataFrame
    stats.loc['Final Portfolio Value'] = [ms_port_vals_is.iloc[-1], sl_port_vals_is.iloc[-1],
                                          benchmark_portval_normed_is.iloc[-1] * 100000,
                                          ms_port_vals_oos.iloc[-1], sl_port_vals_os.iloc[-1],
                                          benchmark_portval_normed_oos.iloc[-1] * 100000]

    # Save the statistics to a CSV file
    # stats.to_csv('results/ml_vs_sl_vs_benchmark_portfolio_stats.csv')
    # Save the statistics to a text file
    with open('p8_results.txt', 'a') as f:
        f.write("\nExperiment 1 Stats:\n")
        f.write(stats.to_string())


def author():
    return "ycheng345"

if __name__ == "__main__":
    testcode()