import gurobipy as gp
import numpy as np
from itertools import product, combinations
from network.util import dist, current_edge, class_indices, lane_indices


def solve_online(instance, positions=None, gap=0.0, timelimit=0, consolelog=False, logfile=None):
    """Solve a network scheduling problem as a MILP.

    The `positions` dictionary maps vehicles to positions for the re-optimization setting.
    Velocities are implicit, because we assume instantaneous acceleration and v_max=1.
    """
    env = gp.Env(empty=True)
    if not consolelog:
        env.setParam('LogToConsole', 0)  # disable console logging
    if logfile is not None:
        env.setParam('LogFile', logfile)
        # make sure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(logfile)), exist_ok=True)
    if timelimit > 0:
        env.setParam('TimeLimit', timelimit)

    env.start()
    g = gp.Model(env=env)

    G = instance['G']
    route = instance['route']
    length = instance['length']
    switch = instance['switch']  # switch-over time

    indices = positions.keys()
    N = class_indices(indices)
    n = lane_indices(indices)

    # big-M
    M = 1000

    y = {}

    if positions is None:
        positions = {}

    # calculate the number of nodes already crossed by each vehicle
    nodes_crossed = {}
    # current edge (v, w)
    edges = {}
    # "edge position" in (v, w)
    edge_pos = {}
    # distance towards next uncrossed node = dist(v, w) - "edge position"
    distance = {}
    for l, k in indices:
        route_pos = positions.get((l, k), 0)
        u, v, pos = current_edge(G, route[l], route_pos)
        edge_pos[l, k] = pos
        distance[l, k] = dist(G, u, v) - pos # distance towards v
        edges[l, k] = (u, v)
        if route_pos > 0:
            nodes_crossed[l, k] = route[l].index(u) + 1
        else:
            nodes_crossed[l, k] = 0

    def crossed(l, k, v):
        """Whether vehicle (l, k) has already crossed node v, so whether
        variable y[l, k, v] should be considered."""
        r = route[l].index(v)
        return r < nodes_crossed[l, k]


    # crossing time variables
    for l, k in indices:
        for r in range(len(route[l])):
            v = route[l][r]
            if crossed(l, k, v):
                continue
            # TODO: only obj=1 for last node on route
            if r == 0:
                # no waiting at entrypoint
                y[l, k, v] = 0
            else:
                y[l, k, v] = g.addVar(obj=1, vtype=gp.GRB.CONTINUOUS, name=f"y_{l}_{k}_{v}")


    # conjunctions...
    for l in N:
        for v in route[l][1:]: # ...on all except the first node
            for k in n[l][:-1]:
                # both vehicles already crossed
                if crossed(l, k, v) and crossed(l, k + 1, v):
                    continue

                elif crossed(l, k, v) and not crossed(l, k + 1, v) and edges[l, k][0] == v:
                    t = length[l][k] - edge_pos[l, k]
                    if t > 0:
                        # vehicle (l, k) still occupies the intersection
                        g.addConstr(t <= y[l, k + 1, v])

                # both vehicles still need to cross
                elif not crossed(l, k, v) and not crossed(l, k + 1, v):
                    g.addConstr(y[l, k, v] + length[l][k] <= y[l, k + 1, v])


    # disjunctions at route intersections
    for l1, l2 in combinations(N, 2):
        # intersections of routes is set of "merge points"
        for v in set(route[l1]) & set(route[l2]):
            for k1, k2 in product(n[l1], n[l2]):
                # both vehicles are already past this intersection
                if crossed(l1, k1, v) and crossed(l2, k2, v):
                    continue

                # both vehicles still need to cross => disjunction 1 <-> 2
                elif not crossed(l1, k1, v) and not crossed(l2, k2, v):
                    oc = g.addVar(obj=0, vtype=gp.GRB.BINARY, name=f"o_{l1}_{k1}_{l2}_{k2}")

                    g.addConstr(y[l1, k1, v] + length[l1][k1] + switch <= y[l2, k2, v] + oc * M)
                    g.addConstr(y[l2, k2, v] + length[l2][k2] + switch <= y[l1, k1, v] + (1 - oc) * M)

                elif crossed(l1, k1, v) and not crossed(l2, k2, v) and edges[l1, k1][0] == v:
                    w1 = edges[l2, k2][1]
                    t1 = length[l1][k1] + switch - edge_pos[l1, k1]
                    if t1 > 0:
                        # vehicle 1 still occupies the intersection => conjunction 1 -> 2
                        g.addConstr(t1 <= y[l2, k2, v])

                elif not crossed(l1, k1, v) and crossed(l2, k2, v) and edges[l2, k2][0] == v:
                    w2 = edges[l2, k2][1]
                    t2 = length[l2][k2] + switch - edge_pos[l2, k2]
                    if t2 > 0:
                        # vehicle 2 still occupies the intersection => conjunction 2 -> 1
                        g.addConstr(t2 <= y[l1, k1, v])


    # distances
    for l, k in indices:
        for r in range(len(route[l]) - 1):
            v = route[l][r]
            w = route[l][r + 1]
            if crossed(l, k, w): # => crossed(l, k, v)
                continue # crossed both nodes
            elif crossed(l, k, v) and not crossed(l, k, w):
                # w is the next uncrossed node, which is distance[l][k] away
                # so assuming v_max=1, it takes at least distance[l][k] time
                g.addConstr(y[l, k, w] >= distance[l, k])
            else:
                # both v and w are uncrossed
                g.addConstr(y[l, k, v] + dist(G, v, w) <= y[l, k, w])


    # buffers
    for l in N:
        for r in range(len(route[l]) - 1):
            v = route[l][r]
            w = route[l][r + 1]
            capacity = G.edges[v, w]['capacity']
            if capacity == -1:
                continue
            for k in n[l][:-capacity]:
                if crossed(l, k + capacity, v):
                    continue

                # TODO: specify rho_hat
                rho_hat = length[l][k]

                if crossed(l, k, w):
                    # schedule time (negative, because in the past), assuming
                    # this vehicle has been driving at v_max=1
                    t = - edge_pos[l, k]
                else:
                    t = y[l, k, w]

                g.addConstr(t + rho_hat <= y[l, k + capacity, v])


    ### Solving

    g.ModelSense = gp.GRB.MINIMIZE
    g.Params.MIPGap = gap
    g.update()
    g.optimize()

    y = { k : (v.X if hasattr(v, 'X') else v) for k, v in y.items() }
    return y, g.getObjective().getValue()
