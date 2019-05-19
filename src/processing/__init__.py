from threading import Thread

from src.processing.Analytics import Analytics
from src.processing.DataHolder import DataHolder
from src.processing.Drawing import Drawing
from src.processing.RealTimePredict import RealTimePredict
from src.processing.SeasonalPeriod import SeasonalPeriod
from queue import Queue

import matplotlib.pyplot as plt

from src.processing.Source import Source

if __name__=='__main__':

    q1 = Queue()
    q2 = Queue()

    source = Source(q1)
    source.generateValuesAsync()

    dataStore = DataHolder(q1, q2)
    dataStoreThread = Thread(target=dataStore.storeNewValue)
    dataStoreThread.start()

    seasonalPeriod = SeasonalPeriod(dataStore)
    seasonalPeriodThread = Thread(target=seasonalPeriod.calculatePeriodInWhile)
    seasonalPeriodThread.start()

    predict = RealTimePredict(seasonalPeriod)

    drawing = Drawing(predict, dataStore)
    drawing.run()

    analytics = Analytics(q2)
    analyticsThread = Thread(target=analytics.storeNewValue)
    analyticsThread.start()
    analytics.run()
    plt.show()


