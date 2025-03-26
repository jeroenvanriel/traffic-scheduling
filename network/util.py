import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import colormaps
import math
from operator import itemgetter


def vehicle_indices(instance):
    """Get list of vehicle indices (r, k) from offline instance specification."""
    if 'release' in instance:
        p = instance['release']
    elif 'length' in instance:
        p = instance['length']
    else:
        raise Exception("Need at least one of the 'release' or 'length' lists.")
    N = range(len(p))
    n = [range(len(p[r])) for r in N]
    return [(r, k) for r in N for k in n[r]]


def route_indices(indices):
    """Return list of route indices r."""
    return sorted(set([i[0] for i in indices]))


def order_indices(indices):
    """Return dict of order indices n[r] for each class r, sorted in increasing order."""
    return { route: sorted([k for (r, k) in indices if r == route]) for route in route_indices(indices) }

def routes_at_intersection(instance):
    """Returns dict mapping intersection v to all the route indices that cross v."""
    routes = { v: set() for v in instance['G'].nodes }
    for i, route in enumerate(instance['route']):
        for v in route:
            routes[v].add(i)
    return routes

def indexed_arrivals(instance):
    """Get tuples (r, k, release date) for every vehicle in instance, sorted by
    increasing release date."""
    indices = vehicle_indices(instance)
    N = class_indices(indices)
    n = lane_indices(indices)
    arrivals = [(r, k, instance['release'][r][k]) for r in N for k in n[r]]
    return sorted(arrivals, key=itemgetter(2))


def plot_schedule(instance, y=None):
    height = 0.7 # row height
    y_scale = 0.7 # horizontal scaling
    fig, ax = plt.subplots()
    cmap = colormaps["Paired"] # lane colors

    indices = vehicle_indices(instance)
    nodes = list(instance['G'].nodes)
    nr_nodes = len(nodes)

    rho = instance['length'] / instance['vmax'] # assumption: each vehicle has same length

    if y is None:
        # release times from instance
        for r, k in indices:
            v = instance['route'][r][0]
            y = instance['release'][r][k]
            row = nodes.index(v) + 1
            ax.add_patch(Rectangle((y, nr_nodes-row - height / 2), width=rho, height=height,
                                linewidth=1, facecolor=cmap(r), edgecolor='k'))
    else:
        # crossing times from schedule
        for r, k in indices:
            for v in instance['route'][r]:
                row = nodes.index(v) + 1
                ax.add_patch(Rectangle((y[r,k,v], nr_nodes-row - height / 2), width=rho, height=height,
                                linewidth=1, facecolor=cmap(r), edgecolor='k'))

    ticks = np.arange(nr_nodes)
    # reverse for top to bottom numbering
    labels = np.flip(nodes, axis=0)
    plt.yticks(ticks=ticks, labels=labels)

    plt.autoscale()
    plt.show()


def draw_network(G):
    nx.draw_networkx(G, nx.get_node_attributes(G, 'pos'))
    plt.gca().set_aspect('equal')
    plt.gca().axis('off') # remove box
    fig = plt.gcf()
    plt.show()
    return fig # to allow optinally fig.savefig()


def plot_trajectories(trajectories, dt):
    for r in range(len(trajectories)):
        for k in range(len(trajectories[r])):
            t0 = trajectories[r][k][0]
            ts = len(trajectories[r][k][1])
            t = [t0 + i*dt for i in range(len(trajectories[r][k][1]))]
            plt.plot(t, trajectories[r][k][1])
        plt.show()


def dist(G, v, w):
    """Euclidean distance between nodes v and w in network G."""
    nodes = G.nodes
    return np.linalg.norm(np.array(nodes[v]['pos']) - np.array(nodes[w]['pos']))


def pos_along_route(G, route, node):
    """Get the route position (relative to first node of route) of a particular node of route."""
    pos, i = 0, 0
    while route[i] != node:
        pos += dist(G, route[i], route[i+1])
        i += 1
    return pos


def current_edge(G, route, pos):
    """Find the current edge (u,v) along the route of a vehicle, given its route
    position. Position x is assumed to be on edge (u,v) if x_u < x <= x_v. Also
    returns the route position relative to node u, so x - x_u. We assume that x
    is in the first edge (u,v) of the route whenever x_u <= x <= x_v."""
    if pos == 0: # first edge of the route is [x_u, x_v]
        return route[0], route[1], 0
    # all other routes are (x_u, x_v]
    cum_pos = 0
    v_prev = route[0]
    for v in route[1:]:
        d = dist(G, v_prev, v)
        if cum_pos < pos and pos <= cum_pos + d:
            return v_prev, v, pos - cum_pos
        cum_pos += d
        v_prev = v
    raise Exception("Position is beyond route.")


def get_pos(G, route, pos, vehicle_l, vehicle_w):
    """Get (x, y, angle) for drawing vehicle rectangle."""
    u, v, pos = current_edge(G, route, pos)
    x1, y1 = G.nodes[u]['pos']
    x2, y2 = G.nodes[v]['pos']

    # calculate angle
    v1 = np.array([x2-x1, y2-y1])
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = np.array([1, 0])
    angle = np.sign(y2-y1) * np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) / math.pi * 180

    # go pos units in direction of edge
    pos -= vehicle_l # we want the front of the vehicle as reference
    pos -= vehicle_w / 2 # route position to real position
    length = dist(G, u, v)
    x = x1 + (x2 - x1) / length * pos
    y = y1 + (y2 - y1) / length * pos

    # correct for vehicle width
    nx = (y1-y2) / length * vehicle_w / 2
    ny = (x2-x1) / length * vehicle_w / 2

    return x - nx, y - ny, angle


def draw_vehicles(G, routes, positions, vehicle_l=2, vehicle_w=1, highlight=None, out=None):
    """Draw network with vehicles at their current positions."""
    fig, ax = plt.subplots()

    lineargs = { 'color': 'k', 'linewidth': 0.8 }
    vehicleargs = { 'facecolor': 'darkgrey', 'edgecolor': 'k', 'linewidth': 0.8 }

    # draw the network outline
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
    rects = {}
    for vehicle, position in positions.items():
        route = routes[vehicle[0]]
        x, y, angle = get_pos(G, route, position, vehicle_l, vehicle_w)
        highlightarg = { 'facecolor': 'blue' } if highlight is not None and vehicle in highlight else {}
        rect = Rectangle((x, y), angle=angle, width=vehicle_l, height=vehicle_w, **{**vehicleargs, **highlightarg})
        rects[vehicle] = rect
        ax.add_patch(rect)

    ax.axis('off')
    ax.set_aspect('equal')
    fig.tight_layout(pad=0.05)
    plt.savefig(out) if out is not None else plt.show()

    return rects


def as_dict(list_of_lists):
    """Turns list of lists into dictionary, so key (r,k) maps to list_of_lists[r][k]."""
    N = len(list_of_lists)
    n = [len(list_of_lists[r]) for r in range(N)]
    d = {}
    for r in range(N):
        for k in range(n[r]):
            d[r,k] = list_of_lists[r][k]
    return d
