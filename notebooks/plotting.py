import numpy as np
import gymnasium as gym
import single_intersection_gym
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_instance(instance, out):
    K = int(instance['K'])

    fig, ax = plt.subplots(figsize=(20, 1))

    height = 0.7
    for k in range(K):
        arrivals = instance[f"arrival{k}"]
        lengths = instance[f"length{k}"]

        for arrival, length in np.nditer([arrivals, lengths]):
            ax.add_patch(Rectangle((arrival, k - height / 2), width=length, height=height, linewidth=1, edgecolor='k'))

    #ax.margins(0.05, 0.5)
    ax.autoscale()
    plt.yticks(np.arange(K))
    plt.tight_layout()
    plt.savefig(out)



def plot_partial_schedule(instance, info, out):
    K = int(instance['K'])

    fig, ax = plt.subplots(figsize=(20, 1))

    height = 0.7
    for k in range(K):
        l = info['vehicles_scheduled'][k]
        print('length', l)
        starts = info['start_time'][k][:l]
        lengths = instance[f'length{k}'][:l]

        print(starts)
        print(lengths)

        for start, length in np.nditer([starts, lengths]):
            ax.add_patch(Rectangle((start, k - height / 2), width=length, height=height, linewidth=1, edgecolor='k'))

    #ax.margins(0.05, 0.5)
    ax.autoscale()
    plt.yticks(np.arange(K))
    plt.tight_layout()
    plt.savefig(out)



n_arrivals_per_lane = 4
length = np.full((n_arrivals_per_lane), 1)

arrival0 = np.array([1, 3, 4, 5])
arrival1 = np.array([2, 3, 4, 5])

instance = { 'K': 2, 'arrival0': arrival0, 'length0': length, 'arrival1': arrival1, 'length1': length }


env = gym.make("SingleIntersectionEnv", K=2, instance=instance)

plot_instance(instance, "instance.pdf")

obs, info = env.reset()

terminated = False
step = 1
while not terminated:
    obs, _, terminated, _, info = env.step(1) # always switch lane
    plot_partial_schedule(instance, info, f"step{step}.pdf")
    step += 1
