from itertools import product
from tqdm import tqdm_notebook

import matplotlib.pyplot as plt                  # plots
import pandas as pd                              # tables and data manipulations
import numpy as np                               # vectors and matrices
import statsmodels.tsa.api as smt
import statsmodels.api as sm

from HoltWinters import mean_absolute_percentage_error

class SARIMA:
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

    def plotWithAutocorrelation(self, series=None, lags=None, figsize=(12, 7), style='bmh'):
        if(series == None):
            series = self.series
        """
            Plot time series, its ACF and PACF, calculate Dickey–Fuller test

            y - timeseries
            lags - how many lags to include in ACF, PACF calculation
        """
        if not isinstance(series, pd.Series):
            series = pd.Series(series)

        with plt.style.context(style):
            fig = plt.figure(figsize=figsize)
            layout = (2, 2)
            ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
            acf_ax = plt.subplot2grid(layout, (1, 0))
            pacf_ax = plt.subplot2grid(layout, (1, 1))

            series.plot(ax=ts_ax)
            p_value = sm.tsa.stattools.adfuller(series)[1]
            ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
            smt.graphics.plot_acf(series, lags=lags, ax=acf_ax)
            smt.graphics.plot_pacf(series, lags=lags, ax=pacf_ax)
            plt.tight_layout()
            plt.show()

    def plotDiff(self):
        timeSeriesDiff = self.series.y1 - self.series.y1.shift(24)
        self.plotWithAutocorrelation(timeSeriesDiff[24:], lags=60)
        return timeSeriesDiff

    def plotDoubleDiff(self, timeSeries):
        timeSeriesDiff = timeSeries - timeSeries.shift(1)
        self.plotWithAutocorrelation(timeSeriesDiff[24 + 1:], lags=60)
        return timeSeriesDiff

    def initParamList(self):
        # setting initial values and some bounds for them
        ps = range(2, 5)
        self.d = 1
        qs = range(2, 5)
        Ps = range(0, 2)
        self.D = 1
        Qs = range(0, 2)
        self.s = 24  # season length is still 24

        # creating list with all the possible combinations of parameters
        parameters = product(ps, qs, Ps, Qs)
        self.parameters_list = list(parameters)


    def optimizeSARIMA(self):
        """
            Return dataframe with parameters and corresponding AIC

            parameters_list - list with (p, q, P, Q) tuples
            d - integration order in ARIMA model
            D - seasonal integration order
            s - length of season
        """

        results = []
        best_aic = float("inf")

        print(self.parameters_list, self.d, self.D, self.s)
        for param in tqdm_notebook(self.parameters_list):
            # we need try-except because on some combinations model fails to converge
            try:
                model = sm.tsa.statespace.SARIMAX(self.series.y1, order=(param[0], self.d, param[1]),
                                                  seasonal_order=(param[2], self.D, param[3], self.s)).fit(disp=-1)
            except:
                continue
            aic = model.aic
            # saving best model, AIC and parameters
            if aic < best_aic:
                best_model = model
                best_aic = aic
                best_param = param
            results.append([param, model.aic])

        result_table = pd.DataFrame(results)
        result_table.columns = ['parameters', 'aic']
        # sorting in ascending order, the lower AIC is - the better
        result_table = result_table.sort_values(by='aic', ascending=True).reset_index(drop=True)

        return result_table

    def runBestModel(self, result_table):
        # set the parameters that give the lowest AIC
        p, q, P, Q = result_table.parameters[0]

        self.bestModel = sm.tsa.statespace.SARIMAX(self.series.y1, order=(p, self.d, q),
                                               seasonal_order=(P, self.D, Q, self.s)).fit(disp=-1)
        print(self.bestModel.summary())


    def plotSARIMA(self, n_steps):
        """
            Plots model vs predicted values

            series - dataset with timeseries
            model - fitted SARIMA model
            n_steps - number of steps to predict in the future

        """
        # adding model values
        data = self.series.copy()
        data.columns = ['actual']
        data['arima_model'] = self.bestModel.fittedvalues
        # making a shift on s+d steps, because these values were unobserved by the model
        # due to the differentiating
        data['arima_model'][:self.s + self.d] = np.NaN

        # forecasting on n_steps forward
        forecast = self.bestModel.predict(start=data.shape[0], end=data.shape[0] + n_steps)
        forecast = data.arima_model.append(forecast)
        # calculate error, again having shifted on s+d steps from the beginning
        error = mean_absolute_percentage_error(data['actual'][self.s + self.d:], data['arima_model'][self.s + self.d:])

        plt.figure(figsize=(15, 7))
        plt.title("Mean Absolute Percentage Error: {0:.2f}%".format(error))
        plt.plot(forecast, color='r', label="model")
        plt.axvspan(data.index[-1], forecast.index[-1], alpha=0.5, color='lightgrey')
        plt.plot(data.actual, label="actual")
        plt.legend()
        plt.grid(True)
        plt.show()