
import datetime as dt  		  	   		  		 			  		 			     			  	 
import os  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
import numpy as np  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
import pandas as pd  		  	   		  		 			  		 			     			  	 
from util import get_data, plot_data  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
def compute_portvals(  		  	   		  		 			  		 			     			  	 
    orders,
    start_val=100000,
    commission=9.95,
    impact=0.005,
):
    # this is the function the autograder will call to test your code  		  	   		  		 			  		 			     			  	 
    # NOTE: orders_file may be a string, or it may be a file object. Your  		  	   		  		 			  		 			     			  	 
    # code should work correctly with either input  		  	   		  		 			  		 			     			  	 
    # TODO: Your code here  		  	   		  		 			  		 			     			  	 

    # Create Order book table
    # order_table = pd.read_csv(orders_file, index_col='Date', na_values=['nan'], parse_dates=True)
    order_table = orders.sort_index()
    # print("Order Table:\n", order_table)
    start_date = order_table.index[0]
    end_date = order_table.index[-1]
    order_table['Symbol'] = 'JPM'
    order_table['Order'] = 'SELL'
    order_table.loc[order_table['Shares'] > 0, 'Order'] = 'BUY'
    order_table['Shares'] = order_table['Shares'].abs()
    order_table = order_table.loc[order_table['Shares'] != 0]

    symbols = order_table['Symbol'].unique().tolist()
    price_table = get_data(symbols, pd.date_range(start_date, end_date))
    price_table['Cash'] = pd.Series(1.0, index=price_table.index)
    # print("Price Table:\n", price_table)

    # Create trades Table
    trade_table = pd.DataFrame(0.0, index=price_table.index, columns=price_table.columns)
    # print("Initial Trade Table:\n", trade_table)

    # Create Holding Table
    holdings_table = pd.DataFrame(0.0, index=price_table.index, columns=price_table.columns)
    holdings_table['Cash'] = start_val
    # print("Initial Holding Table:\n", holdings_table)

    # combine the buy and sell in to shares column
    shares = np.where(order_table['Order'].str.upper() == 'BUY', order_table['Shares'], np.where(order_table['Order'].str.upper() == 'SELL', -order_table['Shares'], 0))
    symbols = order_table['Symbol']

    # iterate the order book to update trade table
    for i, date in enumerate(order_table.index):
        symbol = symbols[i]
        trade_shares = shares[i]
        share_cost = price_table.at[date, symbol]
        stock_cost = trade_shares * share_cost * -1
        trade_table.at[date, 'Cash'] += stock_cost
        trade_table.at[date, symbol] += trade_shares
        trade_cost = share_cost * abs(trade_shares) * impact + commission
        trade_table.at[date, 'Cash'] -= trade_cost

    # update holdings tables by cummulative sum of trade table
    holdings_table = trade_table.cumsum()
    holdings_table['Cash'] = holdings_table['Cash'] + start_val
    # print("Holding Table:\n", holdings_table)

    # update value table by multiply the holding with price table
    values_table = holdings_table.mul(price_table)
    # print("Values Table:\n", values_table)

    # calculate portfolio values
    port_vals = values_table.sum(axis=1)
    # print("Portfolio values:\n", port_vals)

    # Save tables to CSV files
    # order_table.to_csv("orders.csv")
    # price_table.to_csv("prices.csv")
    # trade_table.to_csv("trades.csv")
    # holdings_table.to_csv("holdings.csv")
    # values_table.to_csv("values.csv")
    # port_vals.to_csv("portfolio_values.csv", header=True)

    return port_vals

def author():
    return "ycheng345"

# print(compute_portvals(orders_file="./orders/orders-01.csv", start_val=1000000,commission=9.95,impact=0.005))

def test_code():
    """
    Helper function to test code
    """
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./orders/orders-01.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file=of, start_val=sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2008, 6, 1)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [
        0.2,
        0.01,
        0.02,
        1.5,
    ]
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [
        0.2,
        0.01,
        0.02,
        1.5,
    ]

    # Compare portfolio against $SPX
    print(f"Date Range: {start_date} to {end_date}")
    print()
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")
    print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")
    print()
    print(f"Cumulative Return of Fund: {cum_ret}")
    print(f"Cumulative Return of SPY : {cum_ret_SPY}")
    print()
    print(f"Standard Deviation of Fund: {std_daily_ret}")
    print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")
    print()
    print(f"Average Daily Return of Fund: {avg_daily_ret}")
    print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")
    print()
    print(f"Final Portfolio Value: {portvals[-1]}")


if __name__ == "__main__":
    test_code()
