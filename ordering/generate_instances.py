import numpy as np
import pandas
from exact import solve
from util import plot_schedule, lane_order
from tqdm import trange
from collections import namedtuple

rng = np.random.default_rng()


def lane(n, spec):
    length = rng.uniform(spec.min_length, spec.max_length, size=(n))
    if spec.gap2 is not None:
        gaps = rng.uniform(spec.gap1, spec.gap2, size=(n))
    else:
        gaps = rng.exponential(scale=spec.gap1, size=(n))

    shifted = np.roll(length, 1); shifted[0] = 0
    release = np.cumsum(gaps + shifted)

    return release, length


def generate_instance(spec):
    releases, lengths = [], []
    for nl in spec.n:
        r, l = lane(nl, spec)
        releases.append(r); lengths.append(l)

    return {'release': releases, 'length': lengths, 'switch': spec.s }


Spec = namedtuple("Spec", ["train_reps", "test_reps", "n", "s", "gap1", "gap2", "min_length", "max_length"])

specs = {
    #    train test    n     s    gap    rho
    1: Spec(1000, 100, [10,10], 2,  0, 4,  1, 1),
    2: Spec(1000, 100, [15,15], 2,  0, 4,  1, 1),
    3: Spec(1000, 100, [20,20], 2,  0, 4,  1, 1),
    4: Spec(1000, 100, [25,25], 2,  0, 4,  1, 1), # too big for non-licensed Gurobi

    5: Spec(1000, 100, [10,10], 2,  2, None, 1, 1),
    6: Spec(1000, 100, [15,15], 2,  2, None, 1, 1),
}

for i, spec in specs.items():
    train_data = []
    test_data = []
    for _ in trange(spec.train_reps):
        train_data.append(generate_instance(spec))
    for _ in trange(spec.test_reps):
        test_data.append(generate_instance(spec))

    train_data = pandas.DataFrame(zip(train_data), columns=["instance"])
    train_data.to_pickle(f"data/train_{i}.pkl")
    test_data = pandas.DataFrame(zip(test_data), columns=["instance"])
    test_data.to_pickle(f"data/test_{i}.pkl")
