
import glob
import pandas as pd
from pandas import read_csv
from datetime import datetime

# load data
def parse(x):
	return datetime.strptime(x, '%Y %m %d %H %M')

path =r'D:\CS501\project' # use your path
allFiles = glob.glob(path + "/*.csv")
frame = pd.DataFrame()
list_ = []
for file_ in allFiles:
    dataset = read_csv(file_, skiprows=[0,1],  parse_dates = [['Year', 'Month', 'Day', 'Hour', 'Minute']], index_col=0, date_parser=parse)
    list_.append(dataset)
frame = pd.concat(list_)

#dataset = read_csv('13to15.csv', skiprows=[0,1],  parse_dates = [['Year', 'Month', 'Day', 'Hour', 'Minute']], index_col=0, date_parser=parse)
frame.drop([ 'Clearsky DHI', 'Clearsky DNI', 'Clearsky GHI', 'Cloud Type', 'Wind Direction', 'Wind Speed', 'Fill Flag', 'Precipitable Water','Dew Point', 'Pressure', 'Relative Humidity'], axis=1, inplace=True)
# manually specify column names
frame.columns = ['DHI', 'DNI', 'GHI','Temperature', 'Solar Zenith Angle']
# Rearrange the data order and make sure you put the parameter which you want to predict in first cloumn
frame = frame[['Temperature', 'Solar Zenith Angle','DHI', 'DNI', 'GHI']]
# Date as the index
frame.index.name = 'date'

# Check the data 
print(frame.head())
## save to file
frame.to_csv('123.csv')

### Ploting the input 123.csv data value
from pandas import read_csv
from matplotlib import pyplot
# load dataset
dataset = read_csv('123.csv', header=0, index_col=0)
values = dataset.values
# specify columns to plot
groups = [0, 1, 2, 3, 4]
i = 1
# plot each column
pyplot.figure()
for group in groups:
	pyplot.subplot(len(groups), 1, i)
	pyplot.plot(values[:, group])
	pyplot.title(dataset.columns[group], y=0.5, loc='right')
	i += 1
pyplot.show()

