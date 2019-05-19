from threading import Thread

import time

from src.processing.Utils import getSeries


class Source:

    def __init__(self, queue, delay=0.1):
        self.q = queue
        self.delay = delay

    def generateValuesAsync(self):
        writerP = Thread(target=self.generateValues)
        writerP.start()
        return writerP

    def generateValues(self):
        left = 100
        right = left + 1000
        series, oldSeries = getSeries(left, right)

        for value in series[170:]:
            self.q.put(value)
            time.sleep(self.delay)