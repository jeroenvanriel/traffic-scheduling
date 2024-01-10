import numpy as np


seed=31307741687469044381975587942973893579
rng = np.random.default_rng(seed)

instance_params = [
    {
        's': 2,      # switch-over time
        'n': 30,     # number of arrivals
        'lambda': 2, # mean interarrival time
        'L': [1, 3], # platoon range
    },
    {
        's': 2,      # switch-over time
        'n': 30,     # number of arrivals
        'lambda': 3, # mean interarrival time
        'L': [1, 3], # platoon range
    },
    {
        's': 2,      # switch-over time
        'n': 30,     # number of arrivals
        'lambda': 4, # mean interarrival time
        'L': [1, 3], # platoon range
    },
    {
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


    # number of samples from each instance specification
    samples = 100

    for ix, params in enumerate(instance_params):
        for i in range(samples):
            # TODO: currently two lanes are hardcoded, also in the .npz file
            arrival1, length1 = generate_arrivals(params)
            arrival2, length2 = generate_arrivals(params)

            filename = f"./instances/instance_{ix}_{i}.npz"

            np.savez(filename, arrival1=arrival1, length1=length1,
                     arrival2=arrival2, length2=length2, **params)
