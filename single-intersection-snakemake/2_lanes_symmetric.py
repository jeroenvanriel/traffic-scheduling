import numpy as np

seed=31307741687469044381975587942973893579
rng = np.random.default_rng(seed)

def lane(spec):
    """Generate a stream of arrivals for a single lane."""

    length = rng.integers(1, spec['theta'], size=(spec['n']))
    length_shifted = np.roll(length, 1)
    length_shifted[0] = 0

    interarrival = rng.exponential(scale=spec['lambda'], size=(spec['n']))
    arrival = np.cumsum(interarrival + length_shifted)

    return arrival, length

def generate(spec):
    arrival1, length1 = lane(spec)
    arrival2, length2 = lane(spec)

    return {
        "K": 2,
        "arrival1": arrival1, "length1": length1,
        "arrival2": arrival2, "length2": length2,
    }
