import time

from src.processing.Utils import getSeries


class DataHolder:

    def __init__(self, queueIn, queueOut, limit=200, delay=0.1):
        self.delay = delay
        self.q = queueIn
        self.out = queueOut
        self.limit = limit
        self.x = []
        self.y = []
        # self.xForDrawing = Array('i', [], lock=False)
        self.xForDrawing = []
        self.yForDrawing = []
        self.count = 1
        self.seasonal = []

    def storeNewValue(self):
        left = 100
        right = left + 1000
        series, oldSeries = getSeries(left, right)

        ###############################################################
        step = 170
        self.xForDrawing = list(range(step))
        self.yForDrawing = list(series[:step].values)
        self.count += step
        ###############################################################

        while True:
            if self.q.empty() == False:
                value = self.q.get(timeout=0.1)
                # print("store: {}".format(self.xForDrawing))
                self.x.append(self.count)
                self.y.append(value)
                self.xForDrawing.append(self.count)
                self.yForDrawing.append(value)
                self.out.put({'x': self.count, 'y': value})
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