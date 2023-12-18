import unittest
import numpy as np

import gymnasium as gym
import single_intersection_gym

##
# Note that this test requires the single intersection gym environment to be
# installed as a package (using pip install -e ...) in the current python
# environment.
##

class SingleIntersectionGymEnvTest(unittest.TestCase):

    def test_make_env(self):
        def gen():
            rng = np.random.default_rng()

            n_lanes = 2
            n_arrivals = 5

            # interarrival[lane, platoon id] = time after preceding platoon
            interarrival = rng.exponential(scale=5, size=(n_lanes, n_arrivals))

            # length[lane, platoon] = number of vehicles
            platoon_range=[1, 3]
            length = rng.integers(*platoon_range, size=(n_lanes, n_arrivals))

            length_shifted = np.roll(length, 1, axis=1)
            length_shifted[:, 0] = 0

            # arrivals[lane, platoon] = arrival time
            arrival = np.cumsum(interarrival + length_shifted, axis=1)

            return arrival, length

        env = gym.make("SingleIntersectionEnv", platoon_generator=gen)

        env.reset()
        env.step(0)


    def test_observations(self):
        def gen():
            arrivals = np.array([[0, 2, 3], [0, 1, 3]])
            lengths = np.array([[1, 1, 1], [1, 1, 1]])

            return arrivals, lengths

        env = gym.make("SingleIntersectionEnv", platoon_generator=gen, horizon=1, switch_over=2)

        obs, _= env.reset()
        obs, _, _, _, _ = env.step(0) # schedule platoon from lane 0

        # current time t = 1
        lane0_horizon = np.array([ [1, 1] ]) # 2 - t = 1
        lane1_horizon = np.array([ [-1, 1] ]) # 0 - t = -1
        expected_obs = np.stack([lane0_horizon, lane1_horizon])

        self.assertTrue(np.array_equal(obs, expected_obs.flatten()))


    def test_reward(self):
        def gen():
            arrivals = np.array([[0, 2, 3], [0, 1, 3]])
            lengths = np.array([[1, 1, 1], [1, 1, 1]])

            return arrivals, lengths

        env = gym.make("SingleIntersectionEnv", platoon_generator=gen, horizon=1, switch_over=2)

        env.reset()
        _, reward, _, _, _ = env.step(0) # schedule platoon from lane 0

        # current time t = 1
        self.assertEqual(reward, 0)

        _, reward, _, _, _ = env.step(1) # schedule platoon from lane 1

        # current time t = 3 because switch_over = 2
        self.assertEqual(reward, -3)
