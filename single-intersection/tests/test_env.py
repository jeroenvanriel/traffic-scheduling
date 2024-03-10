import unittest
import numpy as np

import gymnasium as gym
import single_intersection_gym

##
# Note that this test requires the single intersection gym environment to be
# installed as a package (using pip install -e env) in the current python
# environment.
##


def static_gen():
    return {
        'arrival0': np.array([0, 2, 3]),
        'length0': np.array([1, 1, 1]),
        'arrival1': np.array([0, 1, 3]),
        'length1': np.array([1, 1, 1]),
    }


def random_gen():
    rng = np.random.default_rng()

    n_arrivals = 20

    # interarrival[lane, platoon id] = time after preceding platoon
    interarrival = rng.exponential(scale=5, size=(n_arrivals))

    # length[lane, platoon] = number of vehicles
    platoon_range=[1, 3] # so 1 or 2
    length = rng.integers(*platoon_range, size=(n_arrivals))

    length_shifted = np.roll(length, 1)
    length_shifted[0] = 0

    # arrivals[lane, platoon] = arrival time
    arrival = np.cumsum(interarrival + length_shifted)

    return { 'arrival0': arrival, 'length0': length, 'arrival1': arrival, 'length1': length }




class SingleIntersectionGymEnvTest(unittest.TestCase):

    tol = 1e-8

    def test_make_env(self):
        # two lanes with same arrival process
        env = gym.make("SingleIntersectionEnv", K=2, instance_generator=random_gen)

        env.reset()
        env.step(0)


    def evaluate_schedule(self, K, instance, schedule, discount_factor):
        total_reward = 0

        for k in range(K):
            for arrival, length, start, end in zip(
                    instance[f'arrival{k}'], instance[f'length{k}'],
                    schedule[f'start_time_{k}'], schedule[f'end_time_{k}']
            ):
                self.assertTrue(abs(length - (end - start)) < self.tol, "End time in schedule is not correct.")

                # minus sign because penalty
                total_reward -= length * (np.exp(- discount_factor * arrival) - np.exp(- discount_factor * start)) / discount_factor

        return total_reward


    def test_reward(self):
        discount_factor = 0.95

        def test(instance):
            env = gym.make("SingleIntersectionEnv",
                           K=2, instance_generator=lambda: instance,
                           horizon=1, switch_over=2, discount_factor=discount_factor)
            observation, info = env.reset()
            total_reward = info['initial_reward']
            done = False
            t = info['sojourn_time']
            while not done:
                _, reward, done, _, info = env.step(1) # no-wait policy

                total_reward += np.exp(- discount_factor * t) * reward
                t += info['sojourn_time']

            schedule = {
                'start_time_0': info['start_time'][0],
                'end_time_0': info['end_time'][0],
                'start_time_1': info['start_time'][1],
                'end_time_1': info['end_time'][1],
            }

            check = self.evaluate_schedule(2, instance, schedule, discount_factor)

            self.assertTrue(abs(total_reward - check) < self.tol)

        test(static_gen())
        test(random_gen())
