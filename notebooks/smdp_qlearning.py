import numpy as np
import random
from collections import defaultdict
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

import single_intersection_gym

rng = np.random.default_rng()

writer = SummaryWriter("runs/")

discount_factor = 0.01
start_a = 1.0
end_a = 0.005
start_e = 1.0
end_e = 0.1
exploration_factor = 0.8
total_time = 1e8


def linear_schedule(start_e: float, end_e: float, duration: float, t: float):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def random_gen():
    n_arrivals = 100

    # interarrival[lane, platoon id] = time after preceding platoon
    interarrival0 = rng.exponential(scale=3, size=(n_arrivals))
    interarrival1 = rng.exponential(scale=3, size=(n_arrivals))

    # arrivals[lane, platoon] = arrival time
    arrival0 = np.cumsum(interarrival0)
    arrival1 = np.cumsum(interarrival1)

    # length[lane, platoon] = number of vehicles
    length = np.full((n_arrivals), 1)

    return { 'arrival0': arrival0, 'length0': length, 'arrival1': arrival1, 'length1': length }


env = gym.make("SingleIntersectionEnv", K=2, instance=random_gen, switch_over=10,
               horizon=0, discount_factor=discount_factor)

Q = defaultdict(lambda: -2000)
freq = defaultdict(int)


global_t = 0
while global_t < total_time:
    state, info = env.reset()
    state = state.tolist()
    total_reward = info['initial_reward']
    t = info['sojourn_time']

    done = False
    while not done:
        epsilon = linear_schedule(start_e, end_e, exploration_factor * total_time, global_t)
        learning_rate = linear_schedule(start_a, end_a, total_time, global_t)

        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            # argmax Q
            action = 0 if Q[*state, 0] > Q[*state, 1] else 1

        next_state, reward, done, _, info = env.step(action)
        next_state = next_state.tolist()
        sojourn_time = info['sojourn_time']

        # update Q-value
        Q[*state, action] = (1 - learning_rate) * Q[*state, action] + learning_rate * (
            reward + np.exp(-discount_factor * sojourn_time) * max( Q[*next_state, 0], Q[*next_state, 1] )
        )

        total_reward += np.exp(-discount_factor * t) * reward
        t += sojourn_time
        freq[*next_state, action] += 1

        state = next_state

    global_t += t
    print('episodic reward: ', total_reward)
    writer.add_scalar("charts/episodic_return", total_reward, global_t)


sorted_Q = dict(sorted(Q.items()))
print(sorted_Q)

sorted_freq = dict(sorted(freq.items()))
print(sorted_freq)

for y in range(10):
    a = Q[0, y, 0]
    b = Q[0, y, 1]
    print(f"y={y}: ", a, b, 0 if a > b else 1)

