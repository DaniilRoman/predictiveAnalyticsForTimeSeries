from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from HoltWinters import *
from RunScripts import run_triple_exponential_smoothing
from TsUtils import *

import numpy as np                               # vectors and matrices
import matplotlib.pyplot as plt                  # plots
import seaborn as sns                            # more plots

from src.mlModels.MLModels import MLModels
from src.processing.Utils import getSeries

sns.set()

from dateutil.relativedelta import relativedelta # working with dates with style
from scipy.optimize import minimize              # for function minimization

import statsmodels.formula.api as smf            # statistics and econometrics
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs

from itertools import product                    # some useful functions

from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error

def fitModel(data, number):
    if number == 1:
        return ExponentialSmoothing(data, seasonal_periods=30, trend='add', seasonal='mul').fit(use_boxcox=True)
    if number == 2:
        return sm.tsa.statespace.SARIMAX(data, order=(0, 0, 1), seasonal_order=(1, 2, 1, 30),
                                         enforce_stationarity=False, enforce_invertibility=False).fit(disp=-1)
    if number == 3:
        # model = MLModels('./data/notebooks/network.csv', 'date', 'r_asn')
        # model.runLenRegAndPlot()
        lr = LinearRegression(normalize=True)
        x_data = []
        y_data = []
        for d in range(6, data.shape[0]):
            x = data.iloc[d - 6:d].ravel()
            y = data.iloc[d]

            x_data.append(x)
            y_data.append(y)
        x = x_data  # np.array(list(range(len(data)))).reshape(-1, 1)
        y = y_data  #np.array(data)
        # print(x)
        # print(y)
        return lr.fit(x, y)

        # model.applyRidgeCVRegularization()
        # return ExponentialSmoothing(data, seasonal_periods=30, trend='add', seasonal='mul').fit(use_boxcox=True)

import warnings
warnings.filterwarnings('ignore')

# series = pd.read_csv('./data/notebooks/network.csv')
series = pd.read_csv('./data/notebooks/network.csv', index_col=['date'], parse_dates=['date'])
left = 375
right = left + 200
predictCount = 50
dataPredict = series.r_asn[right:right+predictCount]
data = series.r_asn[left:right]

import time

start = time.time()
mode = 3
fit2 = fitModel(data, mode)
if mode == 3:
    data = list(range(right, right + predictCount))
    newX = []
    for d in range(6, predictCount):
        newX = data.iloc[d - 6:d].ravel()
    print(newX)
    forecast = fit2.predict(newX).reshape(1, -1)
else:
    forecast = fit2.forecast(predictCount)
model = run_triple_exponential_smoothing(data)
end = time.time()
print("TIME: ", end - start)
# model.plotHoltWinters(data, plot_intervals=False, plot_anomalies=False)
# plt.show()


plt.plot(range(left, right), data.values, 'blue')
# plt.plot(range(right, right+predictCount), dataPredict, 'blue')
print(forecast.reshape(1, -1).astype(int)[0])
plt.plot(list(range(right, right + 1)), list(forecast.reshape(1, -1).astype(int))[0][0], 'red')
plt.axvline(right, color='k', linestyle='--')
plt.show()



