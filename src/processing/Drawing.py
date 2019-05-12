import datetime as dt
from multiprocessing import Process

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from src.processing.DataHolder import DataHolder
from src.processing.RealTimePredict import RealTimePredict

from threading import Thread

import numpy as np


class Drawing:
    count = 0

    def __init__(self, limit=200):
        self.xs = []
        self.ys = []
        self.limit = limit
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.data = DataHolder()

    def __animate(self, i):

        # Drawing.count = Drawing.count + 1

        # if Drawing.count < 100:
            # temp_c = random.randint(10, 100)

            # Add x and y to lists
            # xs.append(dt.datetime.now().strftime('%H:%M:%S.%f'))
            # ys.append(temp_c)

            # xs = np.linspace(0, 2, 1000)
            # ys = np.sin(2 * np.pi * (xs - 0.01 * i))

            # Limit x and y lists to 20 items
            # xs = xs[-20:]
            # ys = ys[-20:]

        x, y = self.data.getXY()
        if(x != None):
            # Draw x and y lists
            self.xs += x
            self.ys += y

            print("ANIMATE: {} ::: len: {}".format(self.xs, len(self.xs)))

            if(len(self.xs) > self.limit):
                self.xs = self.xs[len(x):]
                self.ys = self.ys[len(x):]

            self.ax.clear()
            self.ax.plot(self.xs, self.ys)

            # Format plot
            plt.xticks(rotation=45, ha='right')
            plt.subplots_adjust(bottom=0.30)
            plt.title('TMP102 Temperature over Time')
            plt.ylabel('Temperature (deg C)')

    def run(self):
        ani = animation.FuncAnimation(self.fig, self.__animate, interval=500)
        plt.show()



if __name__ == '__main__':
    drawing = Drawing()
    p = Thread(target=drawing.data.storeNewValue)
    p.start()
    drawing.run()


# left = 398
# right = left + 200


# seasonalPeriod = SeasonalPeriod()
# realTimePredict = RealTimePredict(series, left, right)

# realTimePredict.seasonalPeriod.period = 45

# seasonal, predict = realTimePredict.predict(series)


# в дата холдере лежит ряд, списки х и y которые отрисовывает в цикле __animate (нужно продумать условие чтобы
# если не изменилось, то не отрисовывать) а в самом дата холдере эти списки уже меняются в процесе прихода новых значений
# отрисовка и изменение значений само собой происходит паралельно в разных потоках