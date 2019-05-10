import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from src.processing.SeasonalPeriod import SeasonalPeriod


class RealTimePredict:

    def __init__(self, series, left, right):
        self.series = series
        self.seasonalPeriod = SeasonalPeriod()
        self.left = left
        self.right = right

    def predict(self, series):
        seasonal = seasonal_decompose(series, model='aditive', freq=self.seasonalPeriod.period).seasonal
        periodLeft, periodRight = self.__lastPeriod(seasonal.values)
        periodCount = periodRight - periodLeft

        ########################### REFACTOR
        seasonal = seasonal + abs(min(seasonal))
        ###########################

        # x = np.array(range(self.left, self.right), dtype=np.double)[np.newaxis]
        x = np.array(range(len(series)), dtype=np.double)[np.newaxis]
        y = np.array(seasonal.values, dtype=np.double)[np.newaxis]

        alpha, beta = self.__algebraicLinearRegressionOnAllSubwindows(x, y)
        # выделяем область предикшена(это один период сезонности)
        alpha = alpha[0][:periodCount]
        beta = beta[0][:periodCount]
        new_x = [x[0][:periodCount]]

        new_y = (alpha + beta * np.array(new_x, dtype=np.double))[0]
        # берем в виде предикшена наш последний период сезонности
        #     new_y = seasonal.values[periodLeft:periodRight]
        return seasonal, new_y

    def __lastPeriod(self, seasonal):
        right = len(seasonal)
        left = right
        value = seasonal[right - 1]
        for i in reversed(range(int(right / 2), right - 2)):
            if (seasonal[i] == value):
                left = i
                break
        return left, right

    def __algebraicLinearRegressionOnAllSubwindows(self, x, y):
        s = {
            (0, 0): self.__subwindowSums(np.ones_like(x)),
            (1, 0): self.__subwindowSums(x),
            (0, 1): self.__subwindowSums(y),
            (2, 0): self.__subwindowSums(np.square(x)),
            (1, 1): self.__subwindowSums(np.multiply(x, y)),
            (0, 2): self.__subwindowSums(np.square(y)),
        }

        # print("+++++++++++++++++++++++++++++++++++++++++++++")
        # print((s[(0, 0)] * s[(1, 1)] - s[(1, 0)] * s[(0, 1)]) /
        #       (s[(0, 0)] * s[(2, 0)] - s[(1, 0)] ** 2))
        # print("+++++++++++++++++++++++++++++++++++++++++++++")

        beta = ((s[(0, 0)] * s[(1, 1)] - s[(1, 0)] * s[(0, 1)]) /
                (s[(0, 0)] * s[(2, 0)] - s[(1, 0)] ** 2))

        alpha = (s[(0, 1)] - beta * s[(1, 0)]) / s[(0, 0)]

        return alpha, beta

    def __subwindowSums(self, v):
        return np.cumsum(np.triu(np.tile(v, (len(v), 1))), axis=-1)