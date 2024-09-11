import matplotlib.pyplot as plt
import numpy as np


def visualize(locations, delta, release, lane, y):
    """y[i,j] = departure time of vehicle j from location i"""

    n = len(release)

    colors = ['green', 'blue']

    # generate trajectories for each vehicle
    for j in range(n):
        x = 0

        xs = [ x ]
        ts = [ float(y[0, j]) ]

        for i in range(1, locations + 1):
            x += 1

            arrival = float(y[i-1, j]) + delta
            departure = float(y[i, j])

            xs.append(x)
            xs.append(x)
            ts.append(arrival)
            ts.append(departure)


        # time on x axis
        plt.plot(ts, xs, color=colors[lane[j]])
        plt.xticks(np.arange(max(ts) + 1))
        plt.yticks(np.arange(max(xs) + 1))

    plt.savefig('report/figures/finite-buffer-schedule.pdf')
    plt.show()
