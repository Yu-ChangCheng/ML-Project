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
    impact_values = [0.00, 0.01, 0.02, 0.03]
    colors = ['green', 'blue','orange', 'red']
    labels = [f'Strategy Learner (Impact: {impact})' for impact in impact_values]

    plt.figure(figsize=(12, 8))

    stats_dict = {}

    for impact, color, label in zip(impact_values, colors, labels):
        np.random.seed(seed)
        random.seed(seed)
        strategy_learner = StrategyLearner.StrategyLearner(verbose=False, impact=impact, commission=0.0)
        strategy_learner.add_evidence(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
        trades_is = strategy_learner.testPolicy(symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31),
                                                sv=100000)
        sl_port_vals_is = marketsimcode.compute_portvals(orders=trades_is, start_val=100000, commission=0.0,
                                                         impact=impact)
        sl_port_vals_normed_is = sl_port_vals_is / sl_port_vals_is.iloc[0]
        plt.plot(sl_port_vals_normed_is.index, sl_port_vals_normed_is, label=label, color=color)

        cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = compute_portfolio_stats(sl_port_vals_is)

        # Count trades
        trade_count = len(trades_is[trades_is['Shares'] != 0])

        stats_dict[impact] = {
            'Cumulative Return': cum_ret,
            'Average Daily Return': avg_daily_ret,
            'Standard Deviation of Daily Returns': std_daily_ret,
            'Sharpe Ratio': sharpe_ratio,
            'Final Portfolio Value': sl_port_vals_is.iloc[-1],
            'Trade Count': trade_count
        }

    plt.xlabel('Date')
    plt.ylabel('Normalized Portfolio Value (JPM)')
    plt.title('Different Impact Values on Strategy Learner (In-Sample)')
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig('images/experiment2.png')
    plt.clf()

    # Save stats to CSV
    stats_df = pd.DataFrame.from_dict(stats_dict, orient='index')
    # stats_df.to_csv('results/Exp_2_strategy_learner_stats.csv')
    with open('p8_results.txt', 'a') as f:
        f.write("\nExperiment 2 Stats:\n")
        f.write(stats_df.to_string())

def author():
    return "ycheng345"

if __name__ == "__main__":
    testcode()
