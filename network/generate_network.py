import networkx as nx


def capacity(G, instance):
    """Calculate the capacity of all edges based on the instance parameters."""
    # we need u_max, v_max, vehicle_l
    pass


def generate_grid_network(grid_m, grid_n, distance=10):
    """Generate a grid-network having n rows of m intersections from west to east.
    At the edges of the network we also connect each intersection to an
    inbound/outbound node. Therefore, the total number of nodes is n*m + 2*(n+m)."""
    G = nx.Graph()

    # Generate the nodes.
    # The node in the i'th row and j'th column is identified as (i,j).
    for i in range(grid_m + 2):
        for j in range(grid_n + 2):
            if (i == 0 or i == grid_m+1) and (j == 0 or j == grid_n+1):
                continue # skip the corners
            G.add_node((i,j), pos=(i*distance, j*distance))


    # Now add the edges.
    for i in range(grid_m + 1):
        for j in range(grid_n + 1):
            if i == 0 and j == 0:
                continue # skip first corner

            # only both direction in the "interior"
            # negative capacity = unlimited
            if j != 0:
                G.add_edge((i,j), (i+1,j), capacity=-1)
            if i != 0:
                G.add_edge((i,j), (i,j+1), capacity=-1)

    return G
