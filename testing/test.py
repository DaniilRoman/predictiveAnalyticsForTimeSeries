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


# extractFeatures(pathToCandyFile, "observation_date")
# #
# #
# #
# #
# featuresFile = pathToCandyFile+'_features.csv'
# features = pd.read_csv(featuresFile)
# #
# # for i in features.columns:
# #     print(i)
# #
# # print(len(features))
# #
# selectFeatures(featuresFile)


# def test():
#     return 1, 2, 3

# q, w = test()
#
# print(q, ' ', w)

# tmp = 'test_features.csv'

# print(tmp[:-len('_features.csv')])

import pandas as pd

d = {'col1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'col2': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]}
df = pd.DataFrame(data=d)

minLag = 1

dataSize = df.__len__()

columnNames = df.columns

for i in range(dataSize, dataSize + minLag):
    df.loc[i] = [None for n in range(len(columnNames))]

for name in df.columns:
    for i in range(minLag, 3):
        df["lag_{}_{}".format(name, i)] = df[name].shift(i)


y = df.dropna()["col1"]

df.drop(labels=columnNames, axis=1, inplace=True)

X = df.dropna()


def timeseries_train_test_split(X, y, test_size, minLag):
    """
        Perform train-test split with respect to time series structure
    """

    X_predict = X.iloc[-minLag:]

    X = X.iloc[:-minLag]

    # get the index after which test set starts
    test_index = int(len(X) * (1 - test_size))

    X_train = X.iloc[:test_index]
    y_train = y.iloc[:test_index]
    X_test = X.iloc[test_index:]
    y_test = y.iloc[test_index:]

    return X_train, X_test, y_train, y_test, X_predict

X_train, X_test, y_train, y_test, X_predict = timeseries_train_test_split(X, y, 0.3, minLag)

print(X)
print(y)

print("--------------------")
print(X_train)
print("--------------------")
print(X_test)
print("--------------------")
print(X_predict)
print("--------------------")
print(y_train)
print("--------------------")
print(y_test)








