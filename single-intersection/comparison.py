import numpy as np
from tqdm import tqdm

import gymnasium as gym
import single_intersection_gym

from scheduling.single_intersection import solve

##
# Number of lanes is currently fixed at 2, both in this comparison as in the gym
# environment and exact (gurobi) solver.
##

n_lanes = 2

n_arrivals = 30   # total number of platoons per lane
switch_over = 2   # time between platoons from distinct lanes
horizon = 10      # determines the gym observation
exact_gap = 0.05  # stopping condition for gurobi

seed=31307741687469044381975587942973893579
rng = np.random.default_rng(seed)


def generate_instance(n_arrivals, mean_interarrival=5, platoon_range=[1, 3]):
    length = rng.integers(*platoon_range, size=(n_lanes, n_arrivals))
    length_shifted = np.roll(length, 1, axis=1)
    length_shifted[:, 0] = 0

    interarrival = rng.exponential(scale=mean_interarrival, size=(n_lanes, n_arrivals))
    arrival = np.cumsum(interarrival + length_shifted, axis=1)

    return arrival, length


def rollout(env, policy):
    env.reset()
    total_reward = 0
    terminated = False
    while not terminated:
        _, reward, terminated, _, _ = env.step(policy())
        total_reward += reward

    return total_reward


policies = [
    lambda: 0, # first lane 0, then all of lane 1
    lambda: 1, # alternate lanes
    lambda: rng.integers(2) # completely random
]


# number of problem instances
N = 100

# averages
exact_return = np.empty(N)
dispatch_return = np.empty((N, 3))

for i in tqdm(range(N)):
    arrival, length = generate_instance(n_arrivals,)

    ## solve exact
    _, _, exact_return[i] = solve(n_arrivals, switch_over, arrival, length, gap=exact_gap, log=False)

    ## apply dispatching policies
    env = gym.make('SingleIntersectionEnv',
                    platoon_generator=lambda: (arrival, length),
                    switch_over=switch_over,
                    horizon=horizon)

    for ix, policy in enumerate(policies):
        dispatch_return[i, ix] = rollout(env, policy)


print(f"mean reward exact: {exact_return.mean()}")
print(f"mean reward policies: {dispatch_return.mean(axis=0)}")
