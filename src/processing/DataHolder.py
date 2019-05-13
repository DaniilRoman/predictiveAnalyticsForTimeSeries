# from multiprocessing import Array
from threading import Thread

import pandas as pd
import time

from src.processing import SeasonalPeriod

class DataHolder:

    def __init__(self, queue, limit=200, delay=0.1):
        self.delay = delay
        self.q = queue
        self.limit = limit
        self.x = []
        self.y = []
        # self.xForDrawing = Array('i', [], lock=False)
        self.xForDrawing = []
        self.yForDrawing = []
        self.count = 1
        self.seasonal = []

    def getSeries(self, left, right):
        series = pd.read_csv('../../data/notebooks/network.csv')
        oldSeries = series.r_asn
        series = oldSeries[left:right]
        return series, oldSeries

    def storeNewValue(self):
        left = 100
        right = left + 1000
        series, oldSeries = self.getSeries(left, right)

        ###############################################################
        step = 170
        self.xForDrawing = list(range(step))
        self.yForDrawing = list(series[:step].values)
        self.count += step
        ###############################################################

        while True:
            if self.q.empty() == False:
                value = self.q.get(timeout=0.01)
                # print("store: {}".format(self.xForDrawing))
                self.x.append(self.count)
                self.y.append(value)
                self.xForDrawing.append(self.count)
                self.yForDrawing.append(value)
                self.count += 1
            time.sleep(self.delay)

    def getXY(self):
        # print("get: {}".format(self.xForDrawing[:]))
        if len(self.xForDrawing) != 0:
            for i in range(100):
                x = self.xForDrawing
                y = self.yForDrawing
                lenX = len(x)
                if lenX == len(y):
                    self.yForDrawing = self.yForDrawing[lenX:]
                    self.xForDrawing = self.xForDrawing[lenX:]
                    return x, y
        return None, None