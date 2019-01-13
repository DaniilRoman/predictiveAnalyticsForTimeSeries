import requests
from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters

from Config import input_file, candy_file, pathToCandyFile, features_file
from ExtractFeatures import *
from RunScripts import runTripleExponentialSmoothing
from SARIMA import SARIMA
from SelectFeatures import select_features, selectFeatures
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
#
#
# candySlice = candy['2000-01-01':]
# print(candySlice.head())
# runTripleExponentialSmoothing(candy)


# d = pd.read_csv(pathToCandyFile + '.csv')
# print(d.head())
# timeStr = "observation_date"
#
# # TODO for speed
# d = d.loc[len(d) * 0.8:]
#
#
# time = d[timeStr]
# print(time.head())
# d.drop(timeStr,  axis=1, inplace=True)
# print(d.head())
# d['id'] = 0
# d['time'] = time
# print(d.head())
#
#
# print(extract_features(d, column_id="id", column_sort='time').head())
#
# def getY(data):
#     series = pd.DataFrame(data.copy())
#     newColumnNames = []
#     newOldColumnNames = {}
#     for i in range(len(series.columns)):
#         key = "y" + str(i)
#         newColumnNames.append(key)
#         newOldColumnNames[key] = series.columns[i]
#     series.columns = newColumnNames
#     series = series.y1
#     return series.y1


extractFeatures(pathToCandyFile, "observation_date")
#
#
#
#
featuresFile = pathToCandyFile+'_features.csv'
features = pd.read_csv(featuresFile)
#
# for i in features.columns:
#     print(i)
#
# print(len(features))
#
selectFeatures(featuresFile)















