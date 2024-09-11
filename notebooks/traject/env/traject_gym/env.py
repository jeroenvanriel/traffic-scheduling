import gymnasium as gym
import numpy as np

from dataclasses import dataclass

@dataclass
class Vehicle():
    lane: int
    x: float
    v: float


class TrajectEnv(gym.Env):

    def __init__(self):

        # we have three lanes: 0, 1, 2
        #      1    2
        # 0 -> . -> . ->
        #      .    .

        # observation is position of each vehicle
        # TODO: alternatively a grid-based representation ("image")
        self.observation_space = gym.spaces.Box(low=-1e3, high=1e6, shape=(self.n_lanes * horizon * 2,))

        # action is product of accelerations for each vehicle
        # acceleration in { -2, -1, 0, 1, 2 }
        self.action_space = gym.spaces.MultiDiscrete([5 for _ in range(6)], start=[-2 for _ in range(6)])


        self.vehicles: list[Vehicle] = [
            Vehicle(0, 0, 1),
            Vehicle(0, 5, 1),
            Vehicle(1, 0, 1),
            Vehicle(1, 5, 1),
            Vehicle(2, 0, 1),
            Vehicle(2, 5, 1),
        ]


    def reset(self, seed=None, options=None):
        return observation, info


    def step(self, action):

        for vehicle in self.vehicles:
            pass




        return observation, reward, terminated, False, info

