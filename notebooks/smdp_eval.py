import numpy as np
import gymnasium as gym

import single_intersection_gym

rng = np.random.default_rng()

discount_factor = 0.01
total_time = 1e6

def threshold_policy(state, m):
    if state[1] >= m:
        return 1
    else:
        return 0


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


env = gym.make("SingleIntersectionEnv", K=2, instance=random_gen, switch_over=8,
               horizon=0, discount_factor=discount_factor)


def eval_policy(m):
    episodic_rewards = []

    global_t = 0
    while global_t < total_time:
        state, info = env.reset()
        state = state.tolist()
        total_reward = info['initial_reward']
        t = info['sojourn_time']

        done = False
        while not done:
            action = threshold_policy(state, m)

            next_state, reward, done, _, info = env.step(action)
            next_state = next_state.tolist()
            sojourn_time = info['sojourn_time']

            total_reward += np.exp(-discount_factor * t) * reward
            t += sojourn_time

            state = next_state

        global_t += t
        episodic_rewards.append(total_reward)

    print(sum(episodic_rewards) / len(episodic_rewards))


for m in [0, 1, 2]:
    eval_policy(m)
