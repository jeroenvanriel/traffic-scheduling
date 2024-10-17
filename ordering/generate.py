import numpy as np
import pickle
from exact import solve
from util import lane_order
from tqdm import trange

rng = np.random.default_rng()

def generate_instance():
    max_length = 2
    mean_gap = 2
    n = 10
    s = 2

    def lane():
        length = rng.integers(1, max_length, size=(n))
        gaps = rng.exponential(scale=mean_gap, size=(n))

        shifted = np.roll(length, 1); shifted[0] = 0
        release = np.cumsum(gaps + shifted)

        return release, length

    r1, l1 = lane()
    r2, l2 = lane()
    return {'release': [r1, r2], 'length': [l1, l2], 'switch': s }


def generate_data(N):
    instances, schedules, etas = [], [], []
    for _ in trange(N):
        instance = generate_instance()
        schedule = solve(instance)
        eta = lane_order(instance, schedule)
        instances.append(instance)
        schedules.append(schedule)
        etas.append(eta)
    return instances, schedules, etas


if __name__=="__main__":
    import sys
    n = int(sys.argv[1])
    instances, schedules, etas = generate_data(n)
    with open('data.pkl', 'wb') as file:
        pickle.dump((instances, schedules, etas), file)
