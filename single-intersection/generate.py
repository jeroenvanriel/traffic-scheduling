import numpy as np


seed=31307741687469044381975587942973893579
rng = np.random.default_rng(seed)

instance_params = [
    {
        'K': 2,      # number of lanes
        's': 2,      # switch-over time
        'n': 30,     # number of arrivals
        'lambda': 2, # mean interarrival time
        'L': [1, 3], # platoon range
    },
    {
        'K': 2,      # number of lanes
        's': 2,      # switch-over time
        'n': 30,     # number of arrivals
        'lambda': 3, # mean interarrival time
        'L': [1, 3], # platoon range
    },
    {
        'K': 2,      # number of lanes
        's': 2,      # switch-over time
        'n': 30,     # number of arrivals
        'lambda': 4, # mean interarrival time
        'L': [1, 3], # platoon range
    },
    {
        'K': 2,      # number of lanes
        's': 2,      # switch-over time
        'n': 30,     # number of arrivals
        'lambda': 5, # mean interarrival time
        'L': [1, 3], # platoon range
    },
]


def generate_arrivals(p):
    """Generate a stream of arrivals for a single lane."""

    length = rng.integers(*p['L'], size=(p['n']))
    length_shifted = np.roll(length, 1)
    length_shifted[0] = 0

    interarrival = rng.exponential(scale=p['lambda'], size=(p['n']))
    arrival = np.cumsum(interarrival + length_shifted)

    return arrival, length


if __name__=="__main__":
    import os

    # number of samples from each instance specification
    samples = 100

    # write the instances here
    os.makedirs(os.path.dirname("./instances/"), exist_ok=True)

    for ix, params in enumerate(instance_params):
        for i in range(samples):
            data = {}
            # generate arrivals for each of the lanes
            for k in range(params['K']):
                arrival, length = generate_arrivals(params)
                data[f"arrival{k}"] = arrival
                data[f"length{k}"] =  length

            filename = f"./instances/instance_{ix}_{i}.npz"
            np.savez(filename, **data, **params)
