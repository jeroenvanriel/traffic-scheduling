import numpy as np

seed=31307741687469044381975587942973893579
rng = np.random.default_rng(seed)

def lane(spec):
    """Generate a stream of arrivals for a single lane."""

    n = int(spec['n'])

    length = rng.integers(1, spec['theta'], size=n)
    length_shifted = np.roll(length, 1)
    length_shifted[0] = 0

    # bimodal distribution of interarrival times as mixture
    choice = rng.binomial(1, spec['eta'], size=n)
    scale = choice * spec['lambda1'] + (1 - choice) * spec['lambda2']
    interarrival = rng.exponential(scale=scale)

    arrival = np.cumsum(interarrival + length_shifted)

    return arrival, length

def instance_generator(spec):
    arrival0, length0 = lane(spec)
    arrival1, length1 = lane(spec)

    return {
        'arrival0': arrival0, 'length0': length0,
        'arrival1': arrival1, 'length1': length1,
    }
