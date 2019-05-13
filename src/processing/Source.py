from threading import Thread

import pandas as pd
import time

from src.processing.DataHolder import DataHolder


class Source:

    def __init__(self, queue, delay=0.1):
        self.q = queue
        self.delay = delay

    def getSeries(self, left, right):
        series = pd.read_csv('../../data/notebooks/network.csv')
        oldSeries = series.r_asn
        series = oldSeries[left:right]
        return series, oldSeries

    def generateValuesAsync(self):
        writerP = Thread(target=self.generateValues)
        writerP.start()
        return writerP

    def generateValues(self):
        left = 100
        right = left + 1000
        series, oldSeries = self.getSeries(left, right)

        for value in series[170:]:
            self.q.put(value)
            time.sleep(self.delay)






# def writer(q):
#     for i in range(3):
#         q.put(i)
#     q.put('None')
#
# def reader(q, data):
#     while True:
#         # print(q.empty())
#         value = q.get()
#         data.append(value)
#         print("READER: {}".format(data))
#         if value == 'None':
#             break
#



# if __name__ == '__main__':
#     q = Queue()
#     data = DataHolder()
#     writerP = Thread(target=writer, args=(q,))
#     writerP.start()
#
#     readerP = Thread(target=reader, args=(q, data.x))
#     readerP.start()
#
#     for i in range(100):
#         print("MAIN: {}".format(data.x))


