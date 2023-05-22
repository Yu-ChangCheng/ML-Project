
#import required packages
import pandas as pd
import matplotlib.pyplot as pyplot
from datetime import datetime
from math import sqrt
from sklearn.metrics import mean_squared_error


# format the date
def parse(x):
	return datetime.strptime(x, '%Y %m %d %H %M')
# read csv file and parse date into first column
df = pd.read_csv("D:\CS501\project\VAR\905712_40.45_-86.9_2015.csv", skiprows=[0,1],  parse_dates = [['Year', 'Month', 'Day', 'Hour', 'Minute']], index_col=0, date_parser=parse)

# reminder for the original column
Original_Column=['DHI',	'DNI',	'GHI',	'Clearsky DHI',	'Clearsky DNI',	'Clearsky GHI',	'Cloud Type',	'Dew Point',	'Temperature',	'Pressure',	'Relative Humidity',	'Solar Zenith Angle',	'Precipitable Water',	'Wind Direction',	'Wind Speed',	'Fill Flag']

# define dataframe column name
df.columns = Original_Column

# drop undesired attribute
df.drop(['Clearsky DHI','Clearsky DNI',	'Clearsky GHI','Cloud Type','Fill Flag','Dew Point',	'Pressure',	'Relative Humidity',	'Solar Zenith Angle',	'Wind Direction',	'Wind Speed'], axis=1, inplace=True)

# Rearrange the data order 
data = df[['Temperature', 'Precipitable Water', 'DHI', 'DNI', 'GHI'  ]]

# Date as the index
data.index.name = 'date'
cols = data.columns

# Check the data 
print(data.head())

#check the dtypes
print(data.dtypes)

##creating the train and validation set in here we take the 0.75 of the dataset as train, 0.25 as test
train = data[:int(0.75*(len(data)))]
valid = data[int(0.75*(len(data))):]
#if you want to change the validation set remember to change line 61 as well C[:int("0.75"*(len(data)))+i]



from statsmodels.tsa.vector_ar.var_model import VAR

#fit the model/ train the model
model = VAR(endog=train)
model_fit = model.fit(2)



#let C as all data
C=data.values

L=[]
# In here we look back 48 time steps (24 hrs) [-48:] and predict the 1 time step (0.5 hr) steps=1 ...
# from validate set C[:int(0.75*(len(data)))+i]
for i in range (len(valid)):
    prediction = model_fit.forecast(C[:int(0.75*(len(data)))+i][-48:], steps=1)
    L.append(prediction[0])

#print the training result and correlate matrix
model_fit.plot()
model_fit.plot_acorr()
print(model_fit.summary())

# plot the result and RMSE
pyplot.figure('Temperature')
L2=[i[0] for i in L]
pyplot.plot(L2, label='Prediction temperature')
pyplot.plot(valid['Temperature'].values, label='Real temperature')
pyplot.legend()
pyplot.show()
print('rmse value for Temperature is : ', sqrt(mean_squared_error(valid['Temperature'].values,L2)))

pyplot.figure('Precipitable Water')
L2=[i[1] for i in L]
pyplot.plot(L2, label='Prediction Precipitable Water')
pyplot.plot(valid['Precipitable Water'].values, label='Real Precipitable Water')
pyplot.legend()
pyplot.show()
print('rmse value for Precipitable Water is : ', sqrt(mean_squared_error(valid['Precipitable Water'].values,L2)))

pyplot.figure('DHI')
L2=[i[2] for i in L]
pyplot.plot(L2, label='Prediction DHI')
pyplot.plot(valid['DHI'].values, label='Real DHI')
pyplot.legend()
pyplot.show()
print('rmse value for DHI is : ', sqrt(mean_squared_error(valid['DHI'].values,L2)))

pyplot.figure('DNI')
L2=[i[3] for i in L]
pyplot.plot(L2, label='Prediction DNI')
pyplot.plot(valid['DNI'].values, label='Real DNI')
pyplot.legend()
pyplot.show()
print('rmse value for DNI is : ', sqrt(mean_squared_error(valid['DNI'].values,L2)))

pyplot.figure('GHI')
L2=[i[4] for i in L]
pyplot.plot(L2, label='Prediction GHI')
pyplot.plot(valid['GHI'].values, label='Real GHI')
pyplot.legend()
pyplot.show()
print('rmse value for GHI is : ', sqrt(mean_squared_error(valid['GHI'].values,L2)))