# read instance_specs from generate.py
# load the corresponding instances
#
# solve with exact solver

import numpy as np
from glob import glob
from collections import defaultdict
from tqdm import tqdm

import torch

import gymnasium as gym
import single_intersection_gym

from generate import instance_specs
from exact import solve
from dqn import QNetwork


seed=31307741687469044381975587942973893579
rng = np.random.default_rng(seed)


def load_instance(path):
    p = np.load(path)

    switch = p['s'] # switch-over time

    # list of K arrays
    release = []
    length = []
    for k in range(p['K']):
        release.append(p[f"arrival{k}"])
        length.append(p[f"length{k}"])

    return switch, release, length


def static_generators(release, length):
    return [ lambda: (release[k], length[k]) for k in range(len(release)) ]


def evaluate(env, model_path, epsilon=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QNetwork(env.observation_space.shape, env.action_space.n).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    obs, _ = env.reset()
    total_reward = 0
    terminated = False
    while not terminated:
        if rng.random() < epsilon:
            action = env.single_action_space.sample()
        else:
            q_values = model(torch.Tensor(obs).to(device))
            action = torch.argmax(q_values).cpu().numpy()
        next_obs, reward, terminated, _, _ = env.step(action)
        total_reward += reward
        obs = next_obs

    return total_reward


if __name__ == "__main__":

    gap = 0.05

    exactscores = defaultdict(list)
    dqnscores = defaultdict(list)

    for ix, spec in enumerate(instance_specs):

        print(f"instance spec {ix}")

        instances = glob(f"instances/instance_{ix}_*.npz")
        for i in tqdm(range(len(instances))):
            switch, release, length = load_instance(f"instances/instance_{ix}_{i}.npz")

            # solve exact
            y, o, obj = solve(switch, release, length, gap=gap, log=False)
            exactscores[ix].append(obj)


            # solve using dqn policy
            generators = static_generators(release, length)
            env = gym.make("SingleIntersectionEnv", platoon_generators=generators, switch_over=switch)
            model_path = f"runs/{ix}__dqn/dqn.cleanrl_model"
            obj = evaluate(env, model_path)
            dqnscores[ix].append(obj)

        print(f"instance {ix}: exact mean obj = {np.array(exactscores[ix]).mean()} (gap = {gap})")
        print(f"instance {ix}: dqn mean obj = {-np.array(dqnscores[ix]).mean()}")
