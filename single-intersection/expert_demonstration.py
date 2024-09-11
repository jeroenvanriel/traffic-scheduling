import numpy as np
import gymnasium as gym
import single_intersection_gym

import heapq


def expert_demonstration(instance, solution):
    # number of lanes
    K = instance['K']

    # derive the global linear order of vehicles from the solution by labeling
    # each time with the lane number and then sorting by lane (using a heap)
    times = [(float(t), k) for k in range(K) for t in solution[f'start_time_{k}']]
    heapq.heapify(times)

    # replay the step-by-step scheduling to record state-action pairs
    sa = []
    env = gym.make("SingleIntersectionEnv", K=K, instance=instance)
    obs, info = env.reset()

    lane = 0 # by definition of the environment
    start = 0 # current start time might be larger, but does not matter for the algorithm

    terminated = False
    while not terminated:
        t = info['current_time'] # completion time of the last scheduled vehicle

        # determine whether a lane switch happened
        # obtain the lane of upcoming vehicle in the global order
        while start < t:
            start, new_lane = heapq.heappop(times)

        a = int(lane != new_lane) # 1 if lane is different, 0 otherwise
        lane = new_lane

        sa.append((obs, a)) # record the state-action pair

        obs, _, terminated, _, info = env.step(a)

    # return a list of state-action pairs
    return sa
