import numpy as np
import networkx as nx
import math


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
                G.add_edge((i,j), (i+1,j), capacity=-1, dist=distance)
            if i != 0:
                G.add_edge((i,j), (i,j+1), capacity=-1, dist=distance)

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

    return G, routes


def set_capacity(instance):
    """Compute the maximum number of vehicles that can occupy each lane and add
    as edge attributed. Entry lanes are given unlimited capacity."""
    min_stop_dist = instance['vmax'] * instance['vmax'] / (2 * instance['amax'])
    G = instance['G']
    for v, w in G.edges:
        if G.degree[v] == 1:
            # set infinite capacity on entry lanes
            G.edges[v, w]['capacity'] = -1 # infinite
        else:
            G.edges[v, w]['capacity'] = math.floor((G[v][w]['dist'] - instance['width'] - 2 * min_stop_dist) / instance['length'])


def bimodal_exponential(n, p, lambda1, lambda2):
    rng = np.random.default_rng()
    ps = rng.binomial(1, p, size=(n))
    return ps * rng.exponential(scale=lambda1, size=(n)) + (1-ps) * rng.exponential(scale=lambda2, size=(n))


def generate_simple_instance(G, routes,
                             vehicle_l=4,
                             vehicle_w=2,
                             vmax=1,
                             amax=1,
                             arrivals_per_route=8,
                             split_p=0.3,  # how often to split platoons
                             intra_gap=1,  # mean gap in platoon
                             inter_gap=8): # mean gap between platoons
    """Create a simple instance from generated network for
    prototyping/testing/demonstration."""
    rng = np.random.default_rng()
    def lane(n):
        length = np.repeat(vehicle_l, n)
        gaps = bimodal_exponential(n, split_p, inter_gap, intra_gap)

        shifted = np.roll(length, 1); shifted[0] = 0
        return np.cumsum(gaps + shifted)

    instance = {
        'G': G, 'route': routes,
        'release': [lane(arrivals_per_route) for l in range(len(routes))],
        # the following parameters are identical among all vehicles
        'length': vehicle_l,
        'width': vehicle_w,
        'vmax': vmax,
        'amax': amax,
        'rho': vehicle_l / vmax,
        'sigma': (vehicle_l + vehicle_w) / vmax,
    }
    set_capacity(instance)
    return instance
