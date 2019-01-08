import requests
from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters

from Config import input_file
from SARIMA import SARIMA
from TsUtils import candy, waves
import pandas as pd

# sarima = SARIMA(waves)
# sarima.plotWithAutocorrelation(lags=5)




# waves = pd.read_csv('./data/waves_price.csv')


# series = pd.DataFrame(waves['Close'], index=waves.index, columns=['Close'])
#
# print(waves.head())
# series = extract_features(waves, column_id="Date", column_sort="Close")
# print(series.head())


d = pd.read_csv(input_file, header=None)

print(d.head())

columns = list(d.columns)
columns.pop()
columns.append('target')
d.columns = columns

y = d.target

print(d.target)

print(int(len(d)*0.8))