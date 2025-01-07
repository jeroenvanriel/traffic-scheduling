import numpy as np
import networkx as nx
from util import dist
import math


def set_distance(G):
    """Compute the length of lanes between all pairs of intersections and add as
    edge attributes."""
    for v, w in G.edges:
        G[v][w]['dist'] = dist(G, v, w)


def set_capacity(instance):
    """Compute the maximum number of vehicles that can occupy each lane and add
    as edge attributed."""
    min_stop_dist = instance['vmax'] * instance['vmax'] / (2 * instance['amax'])
    G = instance['G']
    for v, w in G.edges:
        G.edges[v, w]['capacity'] = math.floor((G[v][w]['dist'] - instance['width'] - 2 * min_stop_dist) / instance['length'])


def generate_grid_network(m, n, distance=10):
    """ Generate a grid-network having n rows of m intersections from west to
    east.

    Returns (G, routes), where G is a networkx graph and each route is a list of
    nodes. At the edges of the network, we also connect each intersection to an
    inbound/outbound node. Therefore, the total number of nodes is n*m +
    2*(n+m). Pure intersections nodes are stored in `network.intersections`. We
    generate all straight routes from west to east and from south to north.
    """
    G = nx.DiGraph()
    G.intersections = []

    # Generate the nodes.
    # The node in the i'th row and j'th column is identified as (i,j).
    for i in range(m + 2):
        for j in range(n + 2):
            if (i == 0 or i == m+1) and (j == 0 or j == n+1):
                continue # skip the corners
            G.add_node((i,j), pos=(i*distance, j*distance))
            # collect pure intersections (not inbound/outbound)
            if not (i == 0 or i == m+1) and not (j == 0 or j == n+1):
                G.intersections.append((i,j))


    # Now add the edges.
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 and j == 0:
                continue # skip first corner

            # only both direction in the "interior"
            # negative capacity = unlimited
            if j != 0:
                G.add_edge((i,j), (i+1,j), capacity=-1)
            if i != 0:
                G.add_edge((i,j), (i,j+1), capacity=-1)

    # Generate the routes.
    routes = [[] for _ in range(n+m)]

    for i in range(m + 2):
        for j in range(n + 2):
            # west-east
            if i != 0 and i != m+1:
                routes[i-1].append((i,j))

            # south-north
            if j != 0 and j != n+1:
                routes[m+j-1].append((i,j))

    set_distance(G)

    return G, routes


def generate_simple_instance():
    """Create a simple instance with randomly generated grid network for
    prototyping/testing/demonstration."""
    G, routes = generate_grid_network(2, 1)

    vehicle_l = 2
    vehicle_w = 1
    vmax = 2
    amax = 1

    N = len(routes)
    n = [3 for _ in range(N)]

    gap1 = 1
    gap2 = 3

    rng = np.random.default_rng()

    def lane(n):
        length = np.repeat(vehicle_l, n)
        gaps = rng.uniform(gap1, gap2, size=(n))

        shifted = np.roll(length, 1); shifted[0] = 0
        release = np.cumsum(gaps + shifted)
        return release

    releases = [lane(n[l]) for l in range(N)]

    instance = {
        'G': G,
        'route': routes,
        'release': releases,
        # the following parameters are identical among all vehicles
        'length': vehicle_l,
        'width': vehicle_w,
        'rho': vehicle_l / vmax,
        'sigma': (vehicle_l + vehicle_w) / vmax,
        'vmax': vmax,
        'amax': amax,
    }

    #set_capacity(instance)
    #for v, w in G.edges:
        #G.edges[v, w]['buffer'] = -1 # infinite capacity

    return instance
