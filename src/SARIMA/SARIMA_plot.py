import matplotlib.pyplot as plt                  # plots
import pandas as pd                              # tables and data manipulations
import numpy as np                               # vectors and matrices
import statsmodels.tsa.api as smt
import statsmodels.api as sm

from src.TsUtils import mean_absolute_percentage_error


def plotWithAutocorrelation(series, lags=None, figsize=(12, 7), style='bmh'):
    """
        Plot time series, its ACF and PACF, calculate Dickey–Fuller test

        series - timeseries
        lags - how many lags to include in ACF, PACF calculation
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)

    with plt.style.context(style):
        plt.figure(figsize=figsize)
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
        plt.savefig("../../data/images/SARIMA/" + "candy_autocorelation.png")
        plt.show()

def plotDiff(series):
    timeSeriesDiff = series - series.shift(24)
    plotWithAutocorrelation(timeSeriesDiff[24:], lags=60)
    return timeSeriesDiff

def plotDoubleDiff(series):
    timeSeriesDiff = series - series.shift(1)
    plotWithAutocorrelation(timeSeriesDiff[24 + 1:], lags=60)
    return timeSeriesDiff

def plotSARIMA(series, bestModel, s, d, n_steps):
    """
            Plots model vs predicted values

            series - dataset with timeseries
            model - fitted SARIMA model
            n_steps - number of steps to predict in the future

        """
    # adding model values

    data = pd.DataFrame(series.copy())

    print("copy")
    print(data.head())

    data.columns = ["actual"]

    print("actual")
    print(data.tail())

    data['arima_model'] = bestModel.fittedvalues

    print("arima_model")
    print(data.tail())

    # making a shift on s+d steps, because these values were unobserved by the model
    # due to the differentiating
    data['arima_model'][:s + d] = np.NaN

    print("arima_model")
    print(data.tail())

    # forecasting on n_steps forward
    forecast = bestModel.predict(start=data.shape[0], end=data.shape[0] + n_steps)
    forecast = data.arima_model.append(forecast)
    # calculate error, again having shifted on s+d steps from the beginning

    print("111111111111111111111111111111111111111")
    print(data.tail())
    print("111111111111111111111111111111111111111")

    actual_data = data["actual"][s + d:]
    arima_data = data['arima_model'][s + d:]

    error = mean_absolute_percentage_error(actual_data, arima_data)

    plt.figure(figsize=(15, 7))
    plt.title("Mean Absolute Percentage Error: {0:.2f}%".format(error))
    plt.plot(forecast, color='r', label="model")
    plt.axvspan(data.index[-1], forecast.index[-1], alpha=0.5, color='lightgrey')
    plt.plot(data.actual, label="actual")
    plt.legend()
    plt.grid(True)
    plt.savefig("../../data/images/SARIMA/" + "candy_sarima.png")
    plt.show()









# Отрисовка прогноза
#
# data["arima_model"] = invboxcox(best_model.fittedvalues, lmbda)
# forecast = invboxcox(best_model.predict(start = data.shape[0], end = data.shape[0]+100), lmbda)
# forecast = data.arima_model.append(forecast).values[-500:]
# actual = data.Users.values[-400:]
# plt.figure(figsize=(15, 7))
# plt.plot(forecast, color='r', label="model")
# plt.title("SARIMA model\n Mean absolute error {} users".format(round(mean_absolute_error(data.dropna().Users, data.dropna().arima_model))))
# plt.plot(actual, label="actual")
# plt.legend()
# plt.axvspan(len(actual), len(forecast), alpha=0.5, color='lightgrey')
# plt.grid(True)


