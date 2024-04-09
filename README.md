# Machine Learning for Multivariate Time Series Data Analysis

## Machine Learning for Trading Projects
The projects mainly involved using Decision Tree/ Random Forest model to predict stock prices. 
I evaluates two trading strategies using the stock symbol "JPM" over specific in-sample (2008-2009) and out-of-sample (2010-2011) periods. The strategies compared are a manual rule-based strategy and a machine-learning based strategy using a Random Tree Learner with Bagging.

Key Techniques and Indicators Used:

Bollinger Bands Percentage (BBP), Price to Simple Moving Average (Price/SMA), and Momentum (MM) were employed as technical indicators.
For the manual strategy, rules were set to trigger buy or sell decisions based on these indicators.
The strategy learner used these indicators as inputs but processed them through a tree-based model to determine trading actions without fixed rules.
Experiments and Findings:

Performance Comparison: The manual strategy and strategy learner were both tested against a simple benchmark strategy over both periods. The manual strategy outperformed the benchmark in both periods, particularly excelling during the in-sample period with significantly higher returns and a better Sharpe ratio. The strategy learner performed exceptionally well during the in-sample period but faltered in the out-of-sample period, likely due to overfitting.

Impact of Transaction Costs: An increase in the impact value generally resulted in poorer performance by the strategy learner during the in-sample period, demonstrating fewer trades and lower returns as the transaction cost discouraged frequent trading.

Conclusions:

The manual strategy proved robust, consistently outperforming the benchmark. Its simple rule-based approach was effective across different market conditions.
The strategy learner showed potential in tightly controlled conditions (in-sample) but its adaptability to new data (out-of-sample) was limited, indicating a need for better generalization capabilities.
Increasing transaction costs (impact values) negatively affected the strategy learner's performance, highlighting the importance of optimizing trading costs in algorithmic strategies.

*To Gatech students, please don't copy and turn in as your submission. If any question contact me at ycheng345@gatech.edu


## VAR Model 
To run the model just download the whole folder and change the data source path name in the VAR.py to train and test the model
The projects mainly involved using VAR model to predict solar radiation

## LSTM Model 
To run the model just download the whole folder and change the data source path name in the Sum_all_csv_in_this_file_and_arrange.py and it will generate a training data set and then we can run Main.py to train and test the model
The projects mainly involved using LSTM model to predict solar radiation

