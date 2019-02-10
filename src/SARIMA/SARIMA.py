from itertools import product
from tqdm import tqdm_notebook #for display progress when we fit model

import pandas as pd                              # tables and data manipulations
import statsmodels.api as sm


class SARIMA:
    def __init__(self, series, columnName = None):
        series = pd.DataFrame(series.copy())
        if(columnName == None):
            newColumnNames = []
            self.newOldColumnNames = {}
            for i in range(len(series.columns)):
                key = "y" + str(i)
                newColumnNames.append(key)
                self.newOldColumnNames[key] = series.columns[i]
            series.columns = newColumnNames
            self.series = series.y1
        else:
            self.series = series[columnName]

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
                model = sm.tsa.statespace.SARIMAX(self.series, order=(param[0], self.d, param[1]),
                                                  seasonal_order=(param[2], self.D, param[3], self.s)).fit(disp=-1)
            except Exception as e:
                print(e)
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

        self.bestModel = sm.tsa.statespace.SARIMAX(self.series, order=(p, self.d, q),
                                               seasonal_order=(P, self.D, Q, self.s)).fit(disp=-1)
        print(self.bestModel.summary())
