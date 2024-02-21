import numpy as np
import gymnasium as gym
import torch

from glob import glob
import time, os

from generate import instance_specs
from dqn import train, Args


# Train DQN by sampling from problem distribution and evaluate on saved
# instances. Save evaluation results to disk.


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


def evaluate(env, model, epsilon=0):
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
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


args = Args()

os.makedirs(os.path.dirname("./data/evaluation/"), exist_ok=True)
times = []


# train for each instance distribution
for ix, spec in enumerate(instance_specs):
    print(f"instance {ix}")

    # for three different horizon sizes
    for horizon in [10, 20, 30]:

        run_name = f"{ix}__horizon{horizon}__dqn"

        # train
        start = time.time()
        model = train(run_name, spec['lanes'], spec['s'], horizon, args)
        times.append(time.time() - start)
        print(f"wall times: {times}")

        # evaluate
        instances = glob(f"data/instances/instance_{ix}_*.npz")
        scores = []
        for i in range(len(instances)):
            switch, release, length = load_instance(f"data/instances/instance_{ix}_{i}.npz")
            generators = static_generators(release, length)
            env = gym.make("SingleIntersectionEnv", platoon_generators=generators, switch_over=switch, horizon=horizon)

            scores.append(evaluate(env, model))

        out_file = f"./data/evaluation/{ix}_dqn_horizon{horizon}.npz"
        np.savez(out_file, scores=scores, times=times)
