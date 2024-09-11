import unittest
import numpy as np

import gymnasium as gym
import single_intersection_gym

from expert_demonstration import expert_demonstration


# mock instance and exact solution
instance = {
    'K': 2,
    's': 2,
    'arrival0': np.array([ 0.46701207,  1.8753719 ,  9.90881823, 11.72964231, 13.10683664,
                           15.76559947, 17.62094624, 19.13041471, 26.63416559, 28.07368221]),
    'arrival1': np.array([ 5.28138009,  9.76052592, 13.81988979, 14.8410027 , 17.01440351,
                        19.88135469, 24.38827582, 25.92661483, 27.13329476, 30.11792691]),
    'length0': np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
    'length1': np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    }
exact_solution = {
    'K': 2,
    's': 2,
    'start_time_0': np.array([ 0.46701207,  1.8753719 , 12.76052592, 13.76052592, 14.76052592,
                              15.76559947, 17.62094624, 19.13041471, 33.11792691, 34.11792691]),
    'start_time_1': np.array([ 5.28138009,  9.76052592, 22.13041471, 23.13041471, 24.13041471,
                               25.13041471, 26.13041471, 27.13041471, 28.13041471, 30.11792691]),
}


class SingleIntersectionExpertDemonstrationTest(unittest.TestCase):

    def test_retrieve_expert_demonstration(self):
        K = instance['K']

        # list of state-action pairs
        sa = expert_demonstration(instance, exact_solution)

        # replay them and compare final solution
        env = gym.make("SingleIntersectionEnv", K=K, instance=instance)
        obs, info = env.reset()

        terminated = False
        it = iter(sa)

        while not terminated:
            s, a = next(it)

            # compare observations
            self.assertTrue(np.array_equal(obs, s))

            obs, _, terminated, _, info = env.step(a)

        # compare
        self.assertTrue(np.array_equal(info['start_time'][0], exact_solution['start_time_0']))
        self.assertTrue(np.array_equal(info['start_time'][1], exact_solution['start_time_1']))

