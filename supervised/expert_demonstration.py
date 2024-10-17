import numpy as np
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import single_intersection_gym

import heapq


def expert_demonstration(instance, solution):
    # number of lanes
    K = instance['K']

    # derive the global linear order of vehicles from the solution by labeling
    # each time with the lane number and then sorting by starting time (using a heap)
    times = [(float(t), k) for k in range(K) for t in solution[f'start_time_{k}']]
    heapq.heapify(times)

    # replay the step-by-step scheduling to record state-action pairs
    states, actions = [], []
    env = gym.make("SingleIntersectionEnv", instance=instance)
    env = FlattenObservation(env)
    obs, info = env.reset()

    lane = 0 # by definition of the environment
    start = -1 # not yet started
    n = 1 # number of jobs to advance to deal with exhaustive service

    terminated = False
    while not terminated:
        t = info['current_time'] # completion time of the last scheduled vehicle

        for _ in range(n):
            start, new_lane = heapq.heappop(times)

        # determine whether a lane switch happened
        a = int(lane != new_lane) # 1 if lane is different, 0 otherwise
        lane = new_lane

        # record the state-action pair
        states.append(obs)
        actions.append(a)

        obs, _, terminated, _, info = env.step(a)
        n = info['vehicles_scheduled_step']

    # return a list of state-action pairs
    return states, actions
