import gymnasium as gym
import numpy as np


class SingleIntersectionEnv(gym.Env):

    def __init__(self, K, instance=None, horizon=10, switch_over=2, discount_factor=0.99):
        """
        `K` is the number of lanes

        `instace` is a dictionary of the form
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
        You may also provide a function that produces the instance on every reset.

        `horizon` is the length of the fixed look-ahead window.

        `switch_over` is the switch over time that is necessary between the
        moments of crossing of vehicles from different intersections.
        """

        assert instance is not None, "Instance (generator function) must be provided."
        self.n_lanes = int(K)
        self.horizon = horizon
        self.switch_over = switch_over
        self.discount_factor = discount_factor

        self._instance = instance

        self.observation_space = gym.spaces.Box(low=-1e3, high=np.inf, shape=(self.n_lanes, 2 + horizon * 2))
        self.action_space = gym.spaces.Discrete(2)


    def reset(self, seed=None, options=None):
        self.arrival, self.length, self.n = self._generate_vehicles()

        # action history
        self.action_sequence = []
        # sequence of lanes
        self.lane_sequence = []

        # number of scheduled vehicles
        self.vehicles_scheduled = np.zeros(self.n_lanes, dtype=int)

        # start and end times of scheduled vehicles
        self.start_time = [ np.empty((self.n[i])) for i in range(self.n_lanes) ]
        self.end_time = [ np.empty((self.n[i])) for i in range(self.n_lanes) ]

        ## calculate initial state

        # always start at the first lane
        self.current_lane = 0
        self.completion_time = 0
        # start of the current macro-step
        self.step_start = 0

        # apply exhaustive service
        self.initial_reward = self._exhaustive_service()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info


    def _generate_vehicles(self):
        arrival = [] # arrival times
        length = [] # vehicle lengths
        n = [] # number of arrivals

        res = self._instance() if callable(self._instance) else self._instance

        for k in range(self.n_lanes):
            a, l = res[f"arrival{k}"], res[f"length{k}"]
            assert a.shape == l.shape, "List of arrivals should have same length as list of vehicle lengths."
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
            done = self.vehicles_scheduled == self.n

            # current lane done
            if done[l]:
                return False

            # some vehicle is still ready for service
            if self.arrival[l][self.vehicles_scheduled[l]] <= self.completion_time:
                return True

            # get other lanes starting from the current
            other_lane_indices = np.roll(np.arange(self.n_lanes), - l)[1:]
            # remove the indices of lanes that are done
            other_lane_indices = filter(lambda i: not done[i], other_lane_indices)

            # next arrival is on current lane
            if all(
                self.arrival[l][self.vehicles_scheduled[l]] <= self.arrival[i][self.vehicles_scheduled[i]]
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

        # next vehicle
        i = self.vehicles_scheduled[l]

        # arrival time and length of next vehicle
        arrival = self.arrival[l][i]
        length = self.length[l][i]

        self.start_time[l][i] = max(self.completion_time + s, arrival)
        self.end_time[l][i] = self.start_time[l][i] + length
        prev_completion_time = self.completion_time
        self.completion_time = self.end_time[l][i]

        # calculate penalty
        penalty = 0
        # compute only for vehicles that have not been scheduled and the current one
        for k in range(self.n_lanes):
            for j in range(self.vehicles_scheduled[k], self.n[k]):
                # only consider customers that have already arrived
                if self.arrival[k][j] > self.completion_time:
                    continue

                a = max(self.arrival[k][j], prev_completion_time)
                b = self.completion_time
                if k == l and j == i: # current scheduled vehicle
                    b = self.completion_time - length


                penalty += self.length[k][j] * (
                    np.exp(-self.discount_factor * (a - self.step_start)) - np.exp(-self.discount_factor * (b - self.step_start))
                ) / self.discount_factor

        self.lane_sequence.append(l)
        self.vehicles_scheduled[l] += 1
        self.current_lane = l

        return - penalty


    def step(self, action):
        self.action_sequence.append(action)

        if action == 0:
            l = self.current_lane
        else:
            l = (self.current_lane + 1) % self.n_lanes

        # move to serve next lane, when no more arrivals
        while self.vehicles_scheduled[l] == self.n[l]:
            l = (l + 1) % self.n_lanes

        # record the start of this macro-step
        self.step_start = self.completion_time

        # serve lane once and apply exhaustive service (macro-step)
        total_reward = self._serve_lane(l) + self._exhaustive_service()

        observation = self._get_obs()
        info = self._get_info()

        terminated = (self.vehicles_scheduled == self.n).all()

        return observation, total_reward, terminated, False, info


    def _get_obs(self):
        # count number of vehicles that have arrived, but not yet processed
        queue_lengths = np.sum(np.array(self.arrival) <= self.completion_time, axis=1) - self.vehicles_scheduled
        queue_lengths = queue_lengths.astype(np.float32)

        # number of vehicles still left to schedule
        remaining = self.n - self.vehicles_scheduled
        remaining = remaining.astype(np.float32)

        # pad with zeros
        future_arrivals = np.zeros((self.n_lanes, self.horizon), dtype=np.float32)
        # pad with zeros
        future_lengths = np.zeros((self.n_lanes, self.horizon), dtype=np.float32)

        # calculate the horizon
        for l in range(self.n_lanes):
            done = self.vehicles_scheduled[l] == self.n[l]
            if done:
                continue

            # index of the first remaining arrival
            i = np.argmax(self.arrival[l] >= self.completion_time)

            for j in range(self.horizon):
                if i + j >= self.n[l]: # no more vehicles
                    break
                future_arrivals[l, j] = self.arrival[l][i + j] - self.completion_time
                future_lengths[l, j] = self.length[l][i + j]

        # stack the three features together per lane
        obs = np.hstack([queue_lengths[np.newaxis].transpose(), remaining[np.newaxis].transpose(), future_arrivals, future_lengths])

        # shift the observations to be from perspective of the current lane
        # minus sign because we roll "to the left"
        lane_indices = np.roll(np.arange(self.n_lanes), - self.current_lane)

        return obs[lane_indices]


    def _get_info(self):
        return {
            "sojourn_time": self.completion_time - self.step_start, # of the last transition
            "initial_reward": self.initial_reward,
            "action_sequence": self.action_sequence,
            "lane_sequence": self.lane_sequence,
            "vehicles_scheduled": self.vehicles_scheduled,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }
