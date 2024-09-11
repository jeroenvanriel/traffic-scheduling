import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()

def value(x, r, s=3):
    s = np.full(x.shape, s)

    return (x >= r) * (2*r + s + 1) \
        + (x < r) * (np.maximum(x, s) + np.maximum(r, np.maximum(x, s) + 1 + s))

# calculated expected value for exponential distribution

N = 100000
scale = 2 # average interarrival length
start = 0
stop = 4 # exclusive
step = 0.1

x = np.arange(start, stop, step)
R = rng.exponential(scale, (len(x), N))
x = np.repeat(np.expand_dims(x, axis=1), N, axis=1)
m = value(x, R).mean(axis=1)

plt.plot(x, m)
plt.show()
