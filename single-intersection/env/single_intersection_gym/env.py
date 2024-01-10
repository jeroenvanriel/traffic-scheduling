import gymnasium as gym
import numpy as np


class SingleIntersectionEnv(gym.Env):

    def __init__(self, platoon_generators=None, horizon=10, switch_over=2):
        """`platoon_generators` is a list of functions, each of which should
        produce the arriving platoons for a lane. For each lane i, these
        arrivals are encoded as a tuple (arrivals, lengths), each of which is a
        one-dimensional numpy array of length n_i.

        `horizon` is the length of the fixed look-ahead window.

        `switch_over` is the switch over time that is necessary between the
        moments of crossing of vehicles from different intersections.
        """

        assert platoon_generators is not None, "Platoon generator function must be provided."
        self.n_lanes = len(platoon_generators)
        self.horizon = horizon
        self.switch_over = switch_over

        self._platoon_generators = platoon_generators

        self.observation_space = gym.spaces.Box(low=-1e3, high=1e6, shape=(self.n_lanes * horizon * 2,))
        self.action_space = gym.spaces.Discrete(2)

        self.lane_color = { 0: (255, 0, 0), 1: (0, 255, 0) }
        self.scale = 15


    def reset(self, seed=None, options=None):
        self.current_lane = 0
        self.completion_time = 0

        self.arrival, self.length, self.n = self._generate_platoons()

        # action history
        self.action_sequence = []
        # sequence of lanes
        self.lane_sequence = []

        # number of scheduled platoons
        self.platoons_scheduled = np.zeros(self.n_lanes, dtype=int)

        # start and end times of scheduled vehicles
        self.start_time = [ np.empty((self.n[i])) for i in range(self.n_lanes) ]
        self.end_time = [ np.empty((self.n[i])) for i in range(self.n_lanes) ]

        observation = self._get_obs()
        info = self._get_info()

        return observation, info


    def _generate_platoons(self):
        arrival = [] # arrival times
        length = [] # platoon lengths
        n = [] # number of arrivals

        for generator in self._platoon_generators:
            a, l = generator()
            assert a.shape == l.shape, "List of arrivals should have same length as list of platoon lengths."
            arrival.append(a)
            length.append(l)
            n.append(a.shape[-1])

        return arrival, length, np.array(n)


    def step(self, action):
        if action == 0:
            l = self.current_lane
        else:
            l = (self.current_lane + 1) % self.n_lanes

        # move to serve next lane, when no more arrivals
        while self.platoons_scheduled[l] == self.n[l]:
            l = (l + 1) % self.n_lanes

        # actual switch-over time
        s = self.switch_over if l != self.current_lane else 0

        # next platoon
        i = self.platoons_scheduled[l]

        # arrival time and length (number of vehicles) of next platoon
        arrival = self.arrival[l][i]
        length = self.length[l][i]

        self.start_time[l][i] = max(self.completion_time + s, arrival)
        self.end_time[l][i] = self.start_time[l][i] + length
        self.completion_time = self.end_time[l][i]

        self.action_sequence.append(action)
        self.current_lane = l
        self.lane_sequence.append(l)
        self.platoons_scheduled[l] += 1

        # penalty is total delay of platoon
        reward = - (self.start_time[l][i] - arrival) * length

        observation = self._get_obs()
        info = self._get_info()

        terminated = (self.platoons_scheduled == self.n).all()

        return observation, reward, terminated, False, info


    def _get_obs(self):
        # last axis contains (arrival, length), hence dim=2 features
        obs = np.zeros((self.n_lanes, self.horizon, 2), dtype=np.float32)
        # Note that by initializing these with zeros, it means that the agent
        # will see zeros for arrival time and platoon length once there are no
        # more arrivals on the current lane.

        # shift the observations to be from perspective of the current lane
        # minus sign because we roll "to the left"
        lane_indices = np.roll(np.arange(self.n_lanes), - self.current_lane)

        for ix, l in enumerate(lane_indices):
            i = self.platoons_scheduled[l]
            # so i is the index of the vehicle to be served next

            for j in range(self.horizon):
                if i + j >= self.n[l]: # no more vehicles
                    break
                obs[ix, j, 0] = self.arrival[l][i + j] - self.completion_time
                obs[ix, j, 1] = self.length[l][i + j]

        return obs.flatten()


    def _get_info(self):
        return { "action_sequence": self.action_sequence,
                 "lane_sequence": self.lane_sequence,
                 "platoons_scheduled": self.platoons_scheduled }
