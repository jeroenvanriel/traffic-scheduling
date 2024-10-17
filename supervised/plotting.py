import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_instance(instance, out=None):
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
    if out is not None:
        plt.savefig(out)
    else:
        plt.show()
    plt.close()


def plot_schedule(instance, out=None):
    K = int(instance['K'])

    fig, ax = plt.subplots(figsize=(20, 1))

    height = 0.7
    for k in range(K):
        starts = instance[f'start_time_{k}']
        lengths = instance[f'length{k}']

        for start, length in np.nditer([starts, lengths]):
            ax.add_patch(Rectangle((start, k - height / 2), width=length, height=height, linewidth=1, edgecolor='k'))

    #ax.margins(0.05, 0.5)
    ax.autoscale()
    plt.yticks(np.arange(K))
    plt.tight_layout()
    if out is not None:
        plt.savefig(out)
    else:
        plt.show()
    plt.close()
