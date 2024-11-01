import numpy as np
import networkx as nx
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
            row = nodes.index(v0) + 1
            ax.add_patch(Rectangle((r, nr_nodes-row - height / 2), width=p, height=height,
                                linewidth=1, facecolor=cmap(l), edgecolor='k'))

    # schedule
    for l in range(N):
        for k in range(n[l]):
            for v in instance['route'][l]:
                row = nodes.index(v) + 1
                p = instance['length'][l][k]
                ax.add_patch(Rectangle((y[l,k,v], nr_nodes-row - height / 2), width=p, height=height,
                                linewidth=1, facecolor=cmap(l), edgecolor='k'))


    ticks = np.arange(nr_nodes)
    # reverse for top to bottom numbering
    labels = np.flip(nodes, axis=0)
    plt.yticks(ticks=ticks, labels=labels)

    plt.autoscale()
    plt.show()


def draw_network(G):
    nx.draw_networkx(G, nx.get_node_attributes(G, 'pos'))
    plt.gca().set_aspect('equal')
    plt.show()
    plt.close()


def plot_trajectories(trajectories, dt):
    for l in range(len(trajectories)):
        for k in range(len(trajectories[l])):
            t0 = trajectories[l][k][0]
            ts = len(trajectories[l][k][1])
            t = np.arange(t0, t0 + ts*dt, dt)
            plt.plot(t, trajectories[l][k][1])
        plt.show()
