import unittest
import numpy as np

from buffers import solve
from visualize import visualize


if __name__ == "__main__":

    locations = 5
    delta = 1
    p = 1
    switch = 1

    release = np.array([0, 1, 2, 3, 5])
    lane    = np.array([0, 0, 1, 1, 1])

    y, obj = solve(locations, delta, p, switch, release, lane)

    print(y, obj)

    visualize(locations, delta, release, lane, y)
