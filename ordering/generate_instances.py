import numpy as np
import pickle
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


Spec = namedtuple("Spec", ["reps", "n", "s", "gap1", "gap2", "min_length", "max_length"])

specs = [
    #   reps    n     s    gap    rho
    Spec(10, [10,10], 2,  0, 4,  1, 1),
    Spec(10, [15,15], 2,  0, 4,  1, 1),
    Spec(10, [20,20], 2,  0, 4,  1, 1),
    Spec(10, [25,25], 2,  0, 4,  1, 1),
]

for i, spec in enumerate(specs):
    data = []
    for _ in trange(spec.reps):
        data.append(generate_instance(spec))

    # visualize the first instance
    plot_schedule(data[0], out=f"../report/data/sample_{i+1}.pdf", clean=True, custom_end_time=40)

    filename = f"data_{i+1}.pkl"
    with open(filename, 'wb') as file:
        pickle.dump(data, file)
