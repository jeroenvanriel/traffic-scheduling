import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import math


def animate(instance, trajectories, dt, vehicle_l=2, vehicle_w=1):
    def dist(v, w):
        nodes = instance['G'].nodes
        return np.linalg.norm(np.array(nodes[v]['pos']) - np.array(nodes[w]['pos']))


    def current_edge(l, k, pos):
        """Find the current edge (u,v) along the route of a vehicle, given its
        position from the start of the route. Also returns the position relative
        to node u."""
        cum_pos = 0
        v_prev = instance['route'][l][0]
        for v in instance['route'][l][1:]:
            d = dist(v_prev, v)
            if cum_pos <= pos and pos < cum_pos + d:
                return v_prev, v, pos - cum_pos
            cum_pos += d
            v_prev = v
        return instance['route'][l][-2], instance['route'][l][-1], pos - cum_pos + d


    def get_pos(l, k, pos):
        """Get (x, y, angle) for drawing vehicle rectangle."""
        u, v, pos = current_edge(l, k, pos)
        x1, y1 = G.nodes[u]['pos']
        x2, y2 = G.nodes[v]['pos']

        # calculate angle
        v1 = np.array([x2-x1, y2-y1])
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = np.array([1, 0])
        angle = np.sign(y2-y1) * np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) / math.pi * 180

        # go pos units in direction of edge
        pos -= vehicle_l # we want the front of the vehicle as reference
        length = dist(u, v)
        x = x1 + (x2 - x1) / length * pos
        y = y1 + (y2 - y1) / length * pos

        # correct for vehicle width
        nx = (y1-y2) / length * vehicle_w / 2
        ny = (x2-x1) / length * vehicle_w / 2

        return x - nx, y - ny, angle


    # draw the network outline
    fig, ax = plt.subplots()

    lineargs = { 'color': 'k', 'linewidth': 0.8 }
    vehicleargs = { 'facecolor': 'darkgrey', 'edgecolor': 'k', 'linewidth': 0.8 }

    G = instance['G']

    for u, v, a in G.edges(data=True):
        x1, y1 = G.nodes[u]['pos']
        x2, y2 = G.nodes[v]['pos']

        # extend vehicle_w to both sides
        length = np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))
        # compute the normal by rotating counterclockwise
        nx = (y1-y2) / length * vehicle_w / 2
        ny = (x2-x1) / length * vehicle_w / 2

        # sidelines
        plt.plot([x1 + nx, x2 + nx], [y1 + ny, y2 + ny], **lineargs)
        plt.plot([x1 - nx, x2 - nx], [y1 - ny, y2 - ny], **lineargs)



    # draw vehicles
    N = len(instance['release'])
    n = [len(instance['release'][l]) for l in range(N)]
    rects = [[] for l in range(N)]
    for l in range(N):
        for k in range(n[l]):
            pos = 0 # initial position
            x, y, angle = get_pos(l, k, pos)
            rect = Rectangle((x, y), angle=angle, width=vehicle_l, height=vehicle_w, **vehicleargs)
            rects[l].append(rect)
            ax.add_patch(rect)


    def update(frame):
        for l in range(N):
            for k in range(n[l]):
                t0, trajectory = trajectories[l][k]
                i = frame - int(t0 / dt)
                if i < 0: # not yet started
                    continue
                if i >= len(trajectory): # done
                    rects[l][k].set_visible(False)
                    continue

                pos = trajectory[min(i, len(trajectory) - 1)]
                x, y, angle = get_pos(l, k, pos)
                rects[l][k].set_x(x)
                rects[l][k].set_y(y)
                rects[l][k].set_angle(angle)

    # compute amount of frames required
    frames  = -1
    for l in range(N):
        for k in range(n[l]):
            t0, trajectory = trajectories[l][k]
            frames = max(frames, int(t0 / dt) + len(trajectory))

    ax.axis('off')
    ax.set_aspect('equal')
    fig.tight_layout(pad=0.05)

    return animation.FuncAnimation(fig=fig, func=update, frames=frames, interval=50)
