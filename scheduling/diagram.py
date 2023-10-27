import sys
import numpy as np
import matplotlib.pyplot as plt

from traffic import read_instance, solve


if __name__ == "__main__":
    instance = sys.argv[1]
    n, m, p, s, release, order, distance = read_instance(instance)

    entrypoints = {0, 3}
    exitpoints = {2, 4}

    B = 4

    # add two jobs for lane A
    n += 2
    order.append([0, 1, 2])
    order.append([0, 1, 2])

    # we will set these according to (x1,x2) in the loop below
    release.append(0)
    release.append(0)

    # configure the grid size
    xb, xe = (0, 0.5)
    yb, ye = (5, 2*s + B*p)
    nx, ny = (20, 20)
    xx, yy = np.meshgrid(np.linspace(xb, xe, nx), np.linspace(yb, ye, ny))

    def compute_regime(x1, x2):
        release[-2] = release[-3] + p + x1
        release[-1] = release[-2] + p + x2

        _, S = solve(n, m, p, s, release, order, distance, entrypoints, exitpoints)

        # With "regime", I mean one of the three cases (i), (ii), (iii), from my
        # notes of the "decision diagram". Note that these cases just represent the
        # number of vehicles from (the remainder of) lane A that go before the
        # platoon B.
        regime = {(0,0): 1, (1,0): 2, (1,1): 3}[S[(1, 0, 8)], S[(1, 0, 9)]]
        return regime

    regimes = np.empty((nx, ny))
    for i in range(nx):
        for j in range(ny):
            # need to cast to native data types (otherwise Gurobi complains)
            regimes[j,i] = compute_regime(float(xx[j, i]), float(yy[j, i]))

    # plot the "decision diagram"
    plt.scatter(xx, yy, c=regimes)
    plt.show()
