import numpy as np

n_lanes = 2

seed=31307741687469044381975587942973893579
rng = np.random.default_rng(seed)

instance_params = [
    {
        's': 2,      # switch-over time
        'n': 30,     # number of arrivals
        'theta': 2,  # mean interarrival time
        'L': [1, 3], # platoon range
    },
    {
        's': 2,      # switch-over time
        'n': 30,     # number of arrivals
        'theta': 3,  # mean interarrival time
        'L': [1, 3], # platoon range
    },
    {
        's': 2,      # switch-over time
        'n': 30,     # number of arrivals
        'theta': 4,  # mean interarrival time
        'L': [1, 3], # platoon range
    },
    {
        's': 2,      # switch-over time
        'n': 30,     # number of arrivals
        'theta': 5,  # mean interarrival time
        'L': [1, 3], # platoon range
    },
]

# number of samples from each instance specification
samples = 10


def generate_instance(p):
    length = rng.integers(*p['L'], size=(n_lanes, p['n']))
    length_shifted = np.roll(length, 1, axis=1)
    length_shifted[:, 0] = 0

    interarrival = rng.exponential(scale=p['theta'], size=(n_lanes, p['n']))
    arrival = np.cumsum(interarrival + length_shifted, axis=1)

    return arrival, length


for ix, params in enumerate(instance_params):
    for i in range(samples):
        arrival, length = generate_instance(params)

        filename = f"./instances/instance_{ix}_{i}.npz"

        np.savez(filename, arrival=arrival, length=length, **params)
