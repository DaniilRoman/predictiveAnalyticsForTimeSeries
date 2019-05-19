import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class AnimateTest:
    def __init__(self):
        pass

    def run(self):
        x = np.linspace(0, 15, 100)

        fig2 = plt.figure()

        p2 = fig2.add_subplot(111)

        p2.set_xlim([0, 15])
        p2.set_ylim([0, 100])

        l2, = p2.plot([], [], 'r')

        def gen2():
            j = 0
            while (True):
                yield j
                j += 1

        def run2(c):
            y = c * x
            l2.set_data(x, y)

        ani2 = animation.FuncAnimation(fig2, run2, gen2, interval=1)
        plt.show()