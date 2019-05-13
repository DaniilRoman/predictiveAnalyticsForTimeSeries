import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Drawing:
    count = 0

    def __init__(self, realTimePredict, data):
        self.xs = []
        self.ys = []
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.data = data
        self.limit = self.data.limit
        self.predict = realTimePredict

    def __animate(self, i):
        x, y = self.data.getXY()
        if(x != None):
            self.xs += x
            self.ys += y

            newX = []
            newY = []
            seasonal = []
            seasonalX = []

            if(len(self.xs) > self.limit):
                self.xs = self.xs[len(x):]
                self.ys = self.ys[len(x):]
                seasonal, newY = self.predict.predict(self.ys)
                seasonalX = self.xs

                max = self.xs[-1]
                newX = list(range(max, max+len(newY)))
                newY = list(newY)

            self.ax.clear()
            self.ax.plot(self.xs, self.ys, 'blue')
            self.ax.plot(newX, newY, 'red')
            self.ax.plot(seasonalX, seasonal, 'yellow')
            plt.axvline(self.xs[-1], color='k', linestyle='--')

            plt.xticks(rotation=45, ha='right')
            plt.subplots_adjust(bottom=0.30)
            plt.title('Traffic')
            plt.ylabel('Traffic (times)')

    def run(self):
        ani = animation.FuncAnimation(self.fig, self.__animate, interval=5)
        plt.show()
