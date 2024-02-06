import gurobipy as gp
import numpy as np
from itertools import combinations, product


def common_substring(a, b):
    """Assumption: at most one common substring."""
    res = []
    first = None

    for x in a:
        if x in b:
            # found first match
            first = x
            break

    if first is not None:
        # match elements from first
        for x, y in zip(a[a.index(first):], b[b.index(first):]):
            if x == y:
                res.append(x)
            else:
                break

    return res


def solve(ptime, switch, distance, buffer, route, arrival, gap=0.0, log=True):
    """
    ptime = time between consecutive arrivals from same lane
    switch = switch-over time between vehicles of distinct lanes

    distance = distance matrix defining the topology of the network
               distance zero indicates "no connection"
    buffer = maximum number of vehicles per lane, given as matrix
             must have same non-negative entries as `distance`

    route = list of routes, each route is a list of intersections
    arrival = for each route, a list of ordered arrival times
    """

    env = gp.Env(empty=True)
    if not log:
        # disable console logging and license information
        env.setParam('OutputFlag', 0)

    env.start()
    g = gp.Model(env=env)

    # big-M
    M = 1000

    # number of routes
    R = len(route)

    ### Variables

    # Vehicles are identified as (r, j), with route id "r" and the relative
    # order "j" of their arrival time.

    # non-negative crossing times
    y = {}
    for r in range(R):
        for k in route[r]: # each intersection
            for j in range(len(arrival[r])):
                # TODO: objective only at exitpoints
                y[k, r, j] = g.addVar(obj=1, vtype=gp.GRB.CONTINUOUS, name=f"y_{k}_({r}_{j})")

                # release date constraints
                g.addConstr(y[k, r, j] >= arrival[r][j])

    ### Constraints

    # conjunctions (travel constraints)
    for r in range(R):
        for j in range(len(arrival[r])): # each vehicle
            for n in range(len(route[r]) - 1):
                # consecutive intersections i -> k
                i = route[r][n]
                k = route[r][n + 1]
                g.addConstr(y[i, r, j] + ptime + distance[i, k] <= y[k, r, j])


    # fixed-order disjunctions (same lane)
    for r in range(R):
        for k in route[r]: # each intersection
            for j in range(len(arrival[r]) - 1):
                # each pair of consecutive vehicles on this route
                g.addConstr(y[k, r, j] + ptime <= y[k, r, j + 1])


    # delayed conjuctions
    for r in range(R):
        for n in range(len(route[r]) - 1):
            # for each consecutive pair of machines (i, k)
            i = route[r][n]
            k = route[r][n + 1]
            b = buffer[i][k]
            for j in range(len(arrival[r]) - b):
                g.addConstr(y[k, r, j] <= y[i, r, j + b] + ptime)


    # disjunctions (conflicts)
    o = {}
    # each combination of routes
    for r1, r2 in combinations(range(R), 2):
        p = common_substring(route[r1], route[r2])
        assert len(p) <= 1, "Merging or splitting is not supported."
        if len(p) == 0:
            continue
        i = p[0] # conflict intersection

        # each pair of vehicles from both routes
        for j, l in product(range(len(arrival[r1])), range(len(arrival[r2]))):
            oc = g.addVar(obj=0, vtype=gp.GRB.BINARY, name=f"o_({r1}_{j})_({r2}_{l})")
            o[r1, j, r2, l] = oc

            g.addConstr(y[i, r1, j] + ptime + switch <= y[i, r2, l] + oc * M)
            g.addConstr(y[i, r2, l] + ptime + switch <= y[i, r1, j] + (1 - oc) * M)


    ### Solving

    g.ModelSense = gp.GRB.MINIMIZE
    g.Params.MIPGap = gap
    g.update()
    g.optimize()

    # Somehow, we need to do this inside the current function definition,
    # otherwise the Gurobi variables don't expose the .X attribute anymore.
    return { k : (v.X if hasattr(v, 'X') else v) for k, v in y.items() }, \
            g.getObjective().getValue()


if __name__ == "__main__":
    pass
