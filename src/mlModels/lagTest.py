import pandas as pd                              # tables and data manipulations
import matplotlib.pyplot as plt                  # plots
import numpy as np                               # vectors and matrices

from sklearn.preprocessing import StandardScaler
# import seaborn as sns                            # more plots

from sklearn.linear_model import LassoCV, RidgeCV
# from sklearn.linear_model import LinearRegression
# from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

# from src.mlModels.ExtractFeatures import extractFeatures
from src.TsUtils import mean_absolute_percentage_error
# from src.mlModels.SelectFeatures import selectFeatures


#
# загрузить данные
#
wavesFilePath = "../../data/waves_price.csv"
candyFilePath = "../../data/candy_production.csv"

# date_column_name = 'Date'
# y_column_name = 'Close'
date_column_name = 'observation_date'
y_column_name = 'IPG3113N'
#
data = pd.read_csv(candyFilePath, index_col=[date_column_name])
#
# data['Volume'] = [x.replace(',', '') for x in data['Volume']]
# data['Market Cap'] = [x.replace(',', '') for x in data['Market Cap']]
#
# data = data.loc[data['Market Cap'] != '-',:]

data.rename(columns={y_column_name:'y'}, inplace=True)


#
# крос валидация
#
tscv = TimeSeriesSplit(n_splits=5)


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

min_lag = 6
max_lag = 25
# y_column_name = "Close"

dataSize = data.__len__()

columnNames = data.columns

for i in range(dataSize, dataSize + min_lag):
    data.loc[i] = [None for n in range(len(columnNames))]

for name in data.columns:
    for i in range(min_lag, max_lag):
        data["lag_{}_{}".format(name, i)] = data[name].shift(i)


y = data.dropna().y

data.drop(labels=columnNames, axis=1, inplace=True)

X = data.dropna()


#
# разбиваем на тестовую и трееровочную выборку
#

print("-----------------")
print(X.head())
print("-----------------")
print(y[:5])

X_train, X_test, y_train, y_test, X_predict = timeseries_train_test_split(X, y, 0.3, min_lag)


#
# тренеруем модель
#

scaler = StandardScaler()

# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# lr = LinearRegression()
lr = LassoCV(cv=tscv)
# lr = RidgeCV(cv=tscv)
# lr = XGBRegressor()
lr.fit(X_train, y_train)


def plotModelResults(model, X_train=X_train, X_test=X_test, plot_intervals=False, plot_anomalies=False):
    """
        Plots modelled vs fact values, prediction intervals and anomalies

    """

    prediction = model.predict(X_test)

    plt.figure(figsize=(15, 7))
    plt.plot(prediction, "g", label="prediction", linewidth=2.0)
    plt.plot(y_test.values, label="actual", linewidth=2.0)

    if plot_intervals:
        cv = cross_val_score(model, X_train, y_train,
                             cv=tscv,
                             scoring="neg_mean_absolute_error")
        mae = cv.mean() * (-1)
        deviation = cv.std()

        scale = 1.96
        lower = prediction - (mae + scale * deviation)
        upper = prediction + (mae + scale * deviation)

        plt.plot(lower, "r--", label="upper bond / lower bond", alpha=0.5)
        plt.plot(upper, "r--", alpha=0.5)

        if plot_anomalies:
            anomalies = np.array([np.NaN] * len(y_test))
            anomalies[y_test < lower] = y_test[y_test < lower]
            anomalies[y_test > upper] = y_test[y_test > upper]
            plt.plot(anomalies, "o", markersize=10, label="Anomalies")

    error = mean_absolute_percentage_error(prediction, y_test)
    plt.title("Mean absolute percentage error {0:.2f}%".format(error))
    plt.legend(loc="best")
    plt.tight_layout()
    plt.grid(True)
    plt.savefig("./waves_Predicted_lag.png")


def plotCoefficients(model):
    """
        Plots sorted coefficient values of the model
    """

    coefs = pd.DataFrame(model.coef_, X_train.columns)
    coefs.columns = ["coef"]
    coefs["abs"] = coefs.coef.apply(np.abs)
    coefs = coefs.sort_values(by="abs", ascending=False).drop(["abs"], axis=1)

    plt.figure(figsize=(15, 7))
    coefs.coef.plot(kind='bar')
    plt.grid(True, axis='y')
    plt.hlines(y=0, xmin=0, xmax=len(coefs), linestyles='dashed')
    plt.savefig("./waves_Coef_lag.png")

def plotPredict(model, x_predict):
    predict = model.predict(x_predict)
    plt.figure(figsize=(15, 7))
    plt.plot(predict, "g", label="new_value", linewidth=2.0)



# predict = lr.predict(x_predict)
plt.figure(figsize=(15, 7))
plt.plot(y_test, "g", label="new_value", linewidth=2.0)
prediction = lr.predict(X_test)
plt.plot(prediction, label="actual", linewidth=2.0)
# plt.plot(lr.predict(X_predict), label="pred", linewidth=2.0)

error = mean_absolute_percentage_error(prediction, y_test)
plt.title("Mean absolute percentage error {0:.2f}%".format(error))
plt.legend(loc="best")
plt.tight_layout()
plt.grid(True)

plotModelResults(lr, plot_intervals=True, plot_anomalies=False)
plotPredict(lr, X_predict)
plotCoefficients(lr)
plt.show()


