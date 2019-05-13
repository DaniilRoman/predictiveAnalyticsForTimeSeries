# import matplotlib.pyplot as plt
#
# # piplain для подсчета
# from src.processing.DataHolder import DataHolder
# from src.processing.RealTimePredict import RealTimePredict
# from src.processing.SeasonalPeriod import SeasonalPeriod
#
# left = 398
# right = left + 200
# count = right - left
# dataHolder = DataHolder()
# series, oldSeries = dataHolder.getSeries(left, right)
#
# # seasonalPeriod = SeasonalPeriod()
# realTimePredict = RealTimePredict(series, left, right)
#
# realTimePredict.seasonalPeriod.period = 45
#
# seasonal, predict = realTimePredict.predict(series)
#
# plt.plot(seasonal)
# plt.plot(range(right, right + len(predict)), predict, 'red')
# plt.plot(oldSeries[left:right + len(predict)], 'green')
# plt.axvline(right, color='k', linestyle='--')
# plt.show()
from threading import Thread

from src.processing.DataHolder import DataHolder
from src.processing.Drawing import Drawing
from src.processing.RealTimePredict import RealTimePredict
from src.processing.SeasonalPeriod import SeasonalPeriod
from queue import Queue

from src.processing.Source import Source

if __name__=='__main__':

    q = Queue()
    source = Source(q)
    source.generateValuesAsync()

    dataStore = DataHolder(q)
    dataStoreThread = Thread(target=dataStore.storeNewValue)
    dataStoreThread.start()

    seasonalPeriod = SeasonalPeriod(dataStore)
    seasonalPeriodThread = Thread(target=seasonalPeriod.calculatePeriodInWhile)
    seasonalPeriodThread.start()

    predict = RealTimePredict(seasonalPeriod)

    drawing = Drawing(predict, dataStore)
    drawing.run()


