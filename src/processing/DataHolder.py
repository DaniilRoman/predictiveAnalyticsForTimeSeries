# from multiprocessing import Array

import pandas as pd
import time

class DataHolder:

    def __init__(self):
        self.x = []
        self.y = []
        # self.xForDrawing = Array('i', [], lock=False)
        self.xForDrawing = []
        self.yForDrawing = []
        self.count = 1

    def getSeries(self, left, right):
        series = pd.read_csv('../../data/notebooks/network.csv')
        oldSeries = series.r_asn
        series = oldSeries[left:right]
        return series, oldSeries

    def storeNewValue(self):
        left = 100
        right = left + 1000
        step = 200
        # count = right - left
        series, oldSeries = self.getSeries(left, right)

        # startSeries = series[:step]
        for value in series:
            print("store: {}".format(self.xForDrawing))
            self.x.append(self.count)
            self.y.append(value)
            if len(self.xForDrawing) == 0:
                self.xForDrawing = [self.count]
            else:
                self.xForDrawing.append(self.count)
            self.yForDrawing.append(value)
            self.count += 1
            time.sleep(0.1)

    def getXY(self):

        print("get: {}".format(self.xForDrawing[:]))

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