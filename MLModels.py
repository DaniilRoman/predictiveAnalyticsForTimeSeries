import pandas as pd                              # tables and data manipulations
import matplotlib.pyplot as plt                  # plots
import numpy as np                               # vectors and matrices

from tsfresh.feature_extraction import extract_features
from sklearn.preprocessing import StandardScaler
import seaborn as sns                            # more plots

from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

from HoltWinters import mean_absolute_percentage_error

class MLModels:
    def __init__(self, series):
        series = pd.DataFrame(series.copy())
        newColumnNames = []
        self.newOldColumnNames = {}
        for i in range(len(series.columns)):
            key = "y" + str(i)
            newColumnNames.append(key)
            self.newOldColumnNames[key] = series.columns[i]
        series.columns = newColumnNames
        self.series = series.y1

    def code_mean(data, cat_feature, real_feature):
        """
        Returns a dictionary where keys are unique categories of the cat_feature,
        and values are means over real_feature
        """
        return dict(data.groupby(cat_feature)[real_feature].mean())


    def timeseries_train_test_split(X, y, test_size_percent):
        """
            Perform train-test split with respect to time series structure
        """

        # get the index after which test set starts
        train_size_percent = 1 - test_size_percent
        test_index = int(len(X) * train_size_percent)

        X_train = X.iloc[:test_index]
        y_train = y.iloc[:test_index]
        X_test = X.iloc[test_index:]
        y_test = y.iloc[test_index:]

        return X_train, X_test, y_train, y_test

    def prepareData(series, lag_start, lag_end, test_size, target_encoding=False):
        """
            series: pd.DataFrame
                dataframe with timeseries

            lag_start: int
                initial step back in time to slice target variable
                example - lag_start = 1 means that the model
                          will see yesterday's values to predict today

            lag_end: int
                final step back in time to slice target variable
                example - lag_end = 4 means that the model
                          will see up to 4 days back in time to predict today

            test_size: float
                size of the test dataset after train/test split as percentage of dataset

            target_encoding: boolean
                if True - add target averages to the dataset

        """

        # copy of the initial dataset
        data = pd.DataFrame(series.copy())
        data.columns = ["y"]

        # lags of series
        for i in range(lag_start, lag_end):
            data["lag_{}".format(i)] = data.y.shift(i)

        # datetime features
        data.index = pd.to_datetime(data.index)
        data["hour"] = data.index.hour
        data["weekday"] = data.index.weekday
        data['is_weekend'] = data.weekday.isin([5, 6]) * 1

        if target_encoding:
            # calculate averages on train set only
            test_index = int(len(data.dropna()) * (1 - test_size))
            data['weekday_average'] = list(map(code_mean(data[:test_index], 'weekday', "y").get, data.weekday))
            data["hour_average"] = list(map(code_mean(data[:test_index], 'hour', "y").get, data.hour))

            # drop encoded variables
            data.drop(["hour", "weekday"], axis=1, inplace=True)

        # train-test split
        y = data.dropna().y
        X = data.dropna().drop(['y'], axis=1)
        X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=test_size)

        return X_train, X_test, y_train, y_test


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

    def runLenRegAndPlot(values):
        scaler = StandardScaler()
        X_train, X_test, y_train, y_test = \
            prepareData(values, lag_start=6, lag_end=25, test_size=0.3, target_encoding=True)

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        lr = LinearRegression()
        lr.fit(X_train_scaled, y_train)

        plotModelResults(lr, X_train=X_train_scaled, X_test=X_test_scaled, plot_intervals=True, plot_anomalies=True)
        plotCoefficients(lr)

    def plotCorrelMatrix(X_train):
        plt.figure(figsize=(10, 8))
        sns.heatmap(X_train.corr())

    def applyRidgeCVRegularization():

        ridge = RidgeCV(cv=tscv)
        ridge.fit(X_train_scaled, y_train)

        plotModelResults(ridge,
                         X_train=X_train_scaled,
                         X_test=X_test_scaled,
                         plot_intervals=True, plot_anomalies=True)
        plotCoefficients(ridge)


    def applyLassoCVRegularization():
        lasso = LassoCV(cv=tscv)
        lasso.fit(X_train_scaled, y_train)

        plotModelResults(lasso,
                         X_train=X_train_scaled,
                         X_test=X_test_scaled,
                         plot_intervals=True, plot_anomalies=True)
        plotCoefficients(lasso)

    def applyXGBoost():
        xgb = XGBRegressor()
        xgb.fit(X_train_scaled, y_train)
        return xgb

    def extractFeatures(series):
        series = extract_features(series)
        series.head()