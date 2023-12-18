import gymnasium as gym
import pygame
import numpy as np

##
# Only 2 lanes are currently supported!
##

class SingleIntersectionEnv(gym.Env):

    def __init__(self, platoon_generator=None, n_lanes=2, horizon=10, switch_over=2):
        self.n_lanes = n_lanes
        self.horizon = horizon
        self.switch_over = switch_over

        assert platoon_generator is not None, "Platoon generator function must be provided."
        self._generate_platoons = platoon_generator

        self.observation_space = gym.spaces.Box(low=-1e3, high=1e6, shape=(n_lanes * horizon * 2,))
        self.action_space = gym.spaces.Discrete(2)

        self.lane_color = { 0: (255, 0, 0), 1: (0, 255, 0) }
        self.scale = 15


    def reset(self, seed=None, options=None):
        self.current_lane = 0
        self.completion_time = 0

        self.arrival, self.length = self._generate_platoons()

        assert self.arrival.shape == self.arrival.shape
        self.n_arrivals = self.arrival.shape[-1]

        # action history
        self.action_sequence = []
        # sequence of lanes
        self.lane_sequence = []

        # number of scheduled platoons
        self.platoons_scheduled = np.zeros(self.n_lanes, dtype=int)

        # start and end times of scheduled vehicles
        self.start_time = np.empty((self.n_lanes, self.n_arrivals))
        self.end_time = np.empty((self.n_lanes, self.n_arrivals))

        observation = self._get_obs()
        info = self._get_info()

        return observation, info


    def step(self, action):
        if action == 0:
            l = self.current_lane
        else:
            l = (self.current_lane + 1) % self.n_lanes

        # TODO: (temporary hack) always serve other lane, when no more arrivals
        # (assuming self.lanes == 2)
        if self.platoons_scheduled[l] == self.n_arrivals:
            l = 1 - l

        # actual switch-over time
        s = self.switch_over if l != self.current_lane else 0

        # next platoon
        i = self.platoons_scheduled[l]

        # arrival time and length (number of vehicles) of next platoon
        arrival = self.arrival[l, i]
        length = self.length[l, i]

        self.start_time[l, i] = max(self.completion_time + s, arrival)
        self.end_time[l, i] = self.start_time[l, i] + length
        self.completion_time = self.end_time[l, i]

        self.action_sequence.append(action)
        self.current_lane = l
        self.lane_sequence.append(l)
        self.platoons_scheduled[l] += 1

        # penalty is total delay of platoon
        reward = - (self.start_time[l, i] - arrival) * length

        observation = self._get_obs()
        info = self._get_info()

        terminated = (self.platoons_scheduled == np.full(self.n_lanes, self.n_arrivals)).all()

        return observation, reward, terminated, False, info


    def _get_obs(self):
        # last axis contains (arrival, length), hence dim=2 features
        obs = np.zeros((self.n_lanes, self.horizon, 2), dtype=np.float32)

        # shift the observations to be from perspective of the current lane
        lane_indices = np.roll(np.arange(self.n_lanes), self.current_lane)

        for ix, l in enumerate(lane_indices):
            i = self.platoons_scheduled[l]

            for j in range(self.horizon):
                if i + j >= self.n_arrivals:
                    break
                obs[ix, j, 0] = self.arrival[l, i + j] - self.completion_time
                obs[ix, j, 1] = self.length[l, i + j]

        return obs.flatten()


    def _get_info(self):
        return { "action_sequence": self.action_sequence,
                 "lane_sequence": self.lane_sequence,
                 "platoons_scheduled": self.platoons_scheduled }
