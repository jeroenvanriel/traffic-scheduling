import gymnasium as gym
import numpy as np


class SingleIntersectionEnv(gym.Env):

    def __init__(self, K, instance_generator=None, horizon=10, switch_over=2):
        """
        `K` is the number of lanes

        `instace_generator` is a function that produces
        a dictionary of the form
        {
          'arrival0': arrival0,
          'length0': length0,
          'arrival1': arrival1,
          'length1': length1,
           ...
          'arrivalK-1': arrivalK-1,
          'lengthK-1': lengthK-1,
        }
        where for each lane i, arrivals and lengths are one-dimensional numpy
        array of length n_i.

        `horizon` is the length of the fixed look-ahead window.

        `switch_over` is the switch over time that is necessary between the
        moments of crossing of vehicles from different intersections.
        """

        assert instance_generator is not None, "Instance generator function must be provided."
        self.n_lanes = int(K)
        self.horizon = horizon
        self.switch_over = switch_over

        self._instance_generator = instance_generator

        self.observation_space = gym.spaces.Box(low=-1e3, high=1e6, shape=(self.n_lanes, horizon, 2))
        self.action_space = gym.spaces.Discrete(2)


    def reset(self, seed=None, options=None):
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

        ## calculate initial state

        # always start at the first lane
        self.current_lane = 0
        self.completion_time = 0

        # apply exhaustive service
        self.initial_reward = self._exhaustive_service()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info


    def _generate_platoons(self):
        arrival = [] # arrival times
        length = [] # platoon lengths
        n = [] # number of arrivals

        res = self._instance_generator()

        for k in range(self.n_lanes):
            a, l = res[f"arrival{k}"], res[f"length{k}"]
            assert a.shape == l.shape, "List of arrivals should have same length as list of platoon lengths."
            arrival.append(a)
            length.append(l)
            n.append(a.shape[-1])

        return arrival, length, np.array(n)


    def _exhaustive_service(self):
        total_reward = 0

        # keep serving current lane as long as there are vehicles in the queue,
        # or whenever the next event is in the current queue
        def go_on():
            l = self.current_lane
            done = self.platoons_scheduled == self.n

            # current lane done
            if done[l]:
                return False

            # some vehicle is still ready for service
            if self.arrival[l][self.platoons_scheduled[l]] <= self.completion_time:
                return True

            # get other lanes starting from the current
            other_lane_indices = np.roll(np.arange(self.n_lanes), - l)[1:]
            # remove the indices of lanes that are done
            other_lane_indices = filter(lambda i: not done[i], other_lane_indices)

            # next arrival is on current lane
            if all(
                self.arrival[l][self.platoons_scheduled[l]] <= self.arrival[i][self.platoons_scheduled[i]]
                for i in other_lane_indices
            ):
                return True

            return False

        while go_on():
            total_reward += self._serve_lane(self.current_lane)

        return total_reward


    def _serve_lane(self, l):
        # actual switch-over time
        s = self.switch_over if l != self.current_lane else 0

        # next platoon
        i = self.platoons_scheduled[l]

        # arrival time and length (number of vehicles) of next platoon
        arrival = self.arrival[l][i]
        length = self.length[l][i]

        self.start_time[l][i] = max(self.completion_time + s, arrival)
        self.end_time[l][i] = self.start_time[l][i] + length
        prev_completion_time = self.completion_time
        self.completion_time = self.end_time[l][i]

        # calculate penalty
        penalty = self.completion_time - np.maximum(prev_completion_time, np.minimum(self.completion_time, self.arrival))
        penalty = self.length * penalty
        # count only for vehicles that have to be scheduled
        for k in range(self.n_lanes):
            j = self.platoons_scheduled[k]
            penalty[k][:j] = 0
        penalty = np.sum(penalty)
        # account for the vehicle we just scheduled
        penalty -= length*length

        self.lane_sequence.append(l)
        self.platoons_scheduled[l] += 1
        self.current_lane = l

        return - penalty


    def step(self, action):
        self.action_sequence.append(action)

        if action == 0:
            l = self.current_lane
        else:
            l = (self.current_lane + 1) % self.n_lanes

        # move to serve next lane, when no more arrivals
        while self.platoons_scheduled[l] == self.n[l]:
            l = (l + 1) % self.n_lanes

        # serve lane once and apply exhaustive service
        total_reward = self._serve_lane(l) + self._exhaustive_service()

        observation = self._get_obs()
        info = self._get_info()

        terminated = (self.platoons_scheduled == self.n).all()

        return observation, total_reward, terminated, False, info


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

        return obs


    def _get_info(self):
        return {
            "initial_reward": self.initial_reward,
            "action_sequence": self.action_sequence,
            "lane_sequence": self.lane_sequence,
            "platoons_scheduled": self.platoons_scheduled,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }
