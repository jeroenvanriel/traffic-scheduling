import numpy as np


seed=31307741687469044381975587942973893579
rng = np.random.default_rng(seed)

# theta
arrival_params = [
    {
        'n': 30,     # number of arrivals
        'lambda': 2, # mean interarrival time
        'L': [1, 3], # platoon range
    },
    {
        'n': 30,     # number of arrivals
        'lambda': 3, # mean interarrival time
        'L': [1, 3], # platoon range
    },
    {
        'n': 30,     # number of arrivals
        'lambda': 4, # mean interarrival time
        'L': [1, 3], # platoon range
    },
    {
        'n': 30,     # number of arrivals
        'lambda': 5, # mean interarrival time
        'L': [1, 3], # platoon range
    },
]

# instance specifications
instance_specs = [
    {
        's': 2,
        'lanes': [
            lambda: generate_arrivals(arrival_params[0]),
            lambda: generate_arrivals(arrival_params[0]),
        ]
    },
    {
        's': 2,
        'lanes': [
            lambda: generate_arrivals(arrival_params[1]),
            lambda: generate_arrivals(arrival_params[1]),
        ]
    },
    {
        's': 2,
        'lanes': [
            lambda: generate_arrivals(arrival_params[2]),
            lambda: generate_arrivals(arrival_params[2]),
        ]
    },
    {
        's': 2,
        'lanes': [
            lambda: generate_arrivals(arrival_params[3]),
            lambda: generate_arrivals(arrival_params[3]),
        ]
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

    for ix, spec in enumerate(instance_specs):
        for i in range(samples):
            data = {}
            # generate arrivals for each of the lanes
            for k, lane in enumerate(spec['lanes']):
                arrival, length = lane()
                data[f"arrival{k}"] = arrival
                data[f"length{k}"] =  length

            filename = f"./instances/instance_{ix}_{i}.npz"
            np.savez(filename, **data, s=spec['s'], K=len(spec['lanes']))
