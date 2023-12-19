
n_lanes = 2
n_arrivals = 30   # total number of platoons per lane
mean_interarrival = 5
platoon_range = [1, 3]

seed=31307741687469044381975587942973893579
rng = np.random.default_rng(seed)


def generate_instance():
    length = rng.integers(*platoon_range, size=(n_lanes, n_arrivals))
    length_shifted = np.roll(length, 1, axis=1)
    length_shifted[:, 0] = 0

    interarrival = rng.exponential(scale=mean_interarrival, size=(n_lanes, n_arrivals))
    arrival = np.cumsum(interarrival + length_shifted, axis=1)

    return arrival, length
