import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import colormaps


def plot_schedule(instance, y=None):

    height = 0.7 # row height
    y_scale = 0.7 # horizontal scaling
    fig, ax = plt.subplots()
    cmap = colormaps["tab10"] # lane colors


    N = len(instance['release']) # number of classes
    n = [len(r) for r in instance['release']] # number of arrivals per class

    nodes = list(instance['G'].nodes)
    nr_nodes = len(nodes)

    # instance
    for l in range(N):
        v0 = instance['route'][l][0]
        release, length = instance['release'][l], instance['length'][l]
        for r, p in np.nditer([release, length]):
            ax.add_patch(Rectangle((r, nr_nodes-v0 - height / 2), width=p, height=height,
                                linewidth=1, facecolor=cmap(l), edgecolor='k'))

    # schedule
    for l in range(N):
        for k in range(n[l]):
            for v in instance['route'][l]:
                ax.add_patch(Rectangle((y[l,k,v], nr_nodes-v - height / 2), width=p, height=height,
                                linewidth=1, facecolor=cmap(l), edgecolor='k'))


    ticks = np.arange(nr_nodes)
    # reverse for top to bottom numbering
    labels = np.flip(nodes)
    plt.yticks(ticks=ticks, labels=labels)

    plt.autoscale()
    plt.show()
