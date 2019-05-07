import pandas as pd                              # tables and data manipulations
import matplotlib.pyplot as plt                  # plots
import numpy as np                               # vectors and matrices

from sklearn.preprocessing import StandardScaler
import seaborn as sns                            # more plots

from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

from src.mlModels.ExtractFeatures import extractFeatures
from src.TsUtils import mean_absolute_percentage_error, consoleLog
from src.mlModels.SelectFeatures import selectFeatures

TEST_SIZE = 0.3
N_SPLITS = 5

MIN_LAG = 3
MAX_LAG = 12

def timeseries_train_test_split(X, y, test_size = TEST_SIZE):
    # get the index after which test set starts
    test_index = int(len(X) * (1 - test_size))

    X_train = X.iloc[:test_index]
    y_train = y.iloc[:test_index]
    X_test = X.iloc[test_index:]
    y_test = y.iloc[test_index:]

    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    model = RidgeCV(cv=getCrossValidation())
    # model = LassoCV(cv=getCrossValidation())
    # model = LinearRegression()
    model.fit(X_train, y_train)
    consoleLog(model, "MODEL", direct=True)
    return model

def extractData(path_to_file, date_column_name, y_column_name):
    #
    # загрузить данные
    #

    data = pd.read_csv(path_to_file, index_col=[date_column_name])

    # data['Volume'] = [x.replace(',', '') for x in data['Volume']]
    # data['Market Cap'] = [x.replace(',', '') for x in data['Market Cap']]
    #
    data.rename(columns={y_column_name: 'y'}, inplace=True)
    #
    # data = data.loc[data['Market Cap'] != '-',:]
    consoleLog(data, "DATA")
    #
    # полготовка данных
    #
    for i in range(MIN_LAG, MAX_LAG):
        data["lag_{}".format(i)] = data.y.shift(i)
    consoleLog(data, "Lags")
    return data

def cleanAndSplitData(data):
    #
    # удаление нулов
    #
    y = data.dropna().y
    X = data.dropna().drop(['y'], axis=1)

    return X, y

def getCrossValidation(n_splits = N_SPLITS):
    return TimeSeriesSplit(n_splits=n_splits)

def get_scale_x(X_train, X_test):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled

def predict_and_plot_model_results(model, pathToSave, X_train, X_test, X_test_scaler,
                                   y_train, y_test, plot_intervals=False, plot_anomalies=False, crossValidationFunc = None):
    prediction = model.predict(X_test_scaler)

    plt.figure(figsize=(15, 7))

    # plt.plot(X_train.index, y_train.values, "g", label="actual-train", linewidth=2.0)
    plt.plot(X_test.index, y_test.values, label="actual-test", linewidth=2.0)

    plt.plot(X_test.index, prediction, "r", label="prediction", linewidth=2.0)

    if plot_intervals:
        cv = cross_val_score(model, X_train, y_train,
                             cv=crossValidationFunc,
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

    error = mean_absolute_percentage_error(y_test, prediction)
    plt.title("Mean absolute percentage error {0:.2f}%".format(error))
    plt.legend(loc="best")
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(pathToSave)


def plotCoefficients(model, pathToSave, X_train):
    coefs = pd.DataFrame(model.coef_, X_train.columns)
    coefs.columns = ["coef"]
    coefs["abs"] = coefs.coef.apply(np.abs)
    coefs = coefs.sort_values(by="abs", ascending=False).drop(["abs"], axis=1)

    plt.figure(figsize=(15, 7))
    coefs.coef.plot(kind='bar')
    plt.grid(True, axis='y')
    plt.hlines(y=0, xmin=0, xmax=len(coefs), linestyles='dashed')
    plt.savefig(pathToSave)

########################################################################################################################

def mlPipeline(path_to_file, date_column_name, y_column_name):
    data = extractData(path_to_file, date_column_name, y_column_name)
    X, y = cleanAndSplitData(data)
    X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y)

    X_train_scaler, X_test_scaler = get_scale_x(X_train, X_test)

    model = train_model(X_train_scaler, y_train)

    crossValidationFunc = getCrossValidation()
    predict_and_plot_model_results(model, './test.png', X_train, X_test, X_test_scaler,
                                   y_train, y_test,
                                   plot_intervals=False, crossValidationFunc=crossValidationFunc)
    plotCoefficients(model, './test_2.png', X_train)
    plt.show()


########################################################################################################################

wavesFilePath = "../../data/waves_price.csv"
candyFilePath = "../../data/candy_production.csv"

date_column_name = 'observation_date'
y_column_name = 'IPG3113N'

# date_column_name = 'Date'
# y_column_name = 'Close'

pathToFile = candyFilePath

mlPipeline(pathToFile, date_column_name, y_column_name)

