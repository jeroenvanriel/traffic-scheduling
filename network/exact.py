import gurobipy as gp
import numpy as np
from itertools import product, combinations
from util import dist


def solve(instance, gap=0.0, timelimit=0, consolelog=False, logfile=None):
    """Solve a network scheduling problem as a MILP."""
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

    route = instance['route']
    release = instance['release']

    # for now, assume these are the same among vehicles
    rho = instance['rho']
    sigma = instance['sigma'] # "switch-over" time

    N = len(release) # number of classes
    n = [len(r) for r in release] # number of arrivals per class

    # big-M
    M = 1000

    y = {}

    # release time parameters and crossing time variables
    for l in range(N):
        for k in range(n[l]):
            for r in range(len(route[l])):
                v = route[l][r]
                if r == 0: # entrypoint
                    y[l, k, v] = release[l][k]
                elif r == len(route[l]) - 1: # exitpoint
                    y[l, k, v] = g.addVar(obj=0, vtype=gp.GRB.CONTINUOUS, name=f"y_{l}_{k}_{v}")
                else: # intersections
                    y[l, k, v] = g.addVar(obj=1, vtype=gp.GRB.CONTINUOUS, name=f"y_{l}_{k}_{v}")

    # conjunctions...
    for l in range(N):
        for v in route[l][1:]: # ...on all except the first node
            for k in range(n[l] - 1):
                g.addConstr(y[l, k, v] + rho <= y[l, k + 1, v])


    # disjunctions at route intersections
    for l1, l2 in combinations(range(N), 2):
        # intersections of routes is set of "merge points"
        for v in set(route[l1]) & set(route[l2]):
            for k1, k2 in product(range(n[l1]), range(n[l2])):
                oc = g.addVar(obj=0, vtype=gp.GRB.BINARY, name=f"o_{l1}_{k1}_{l2}_{k2}")

                g.addConstr(y[l1, k1, v] + sigma <= y[l2, k2, v] + oc * M)
                g.addConstr(y[l2, k2, v] + sigma <= y[l1, k1, v] + (1 - oc) * M)


    # distances
    for l in range(N):
        for k in range(n[l]):
            for r in range(len(route[l]) - 1):
                v = route[l][r]
                w = route[l][r + 1]
                g.addConstr(y[l, k, v] + dist(instance['G'], v, w) / instance['vmax'] <= y[l, k, w])

    # buffers
    for l in range(N):
        for r in range(len(route[l]) - 1):
            v = route[l][r]
            w = route[l][r + 1]
            capacity = instance['G'].edges[v, w]['capacity']
            if capacity == -1:
                continue
            for k in range(n[l] - capacity):
                rho_hat = capacity * rho - dist(instance['G'], v, w) / instance['vmax']
                g.addConstr(y[l, k, w] + rho_hat <= y[l, k + capacity, v])

    ### Solving

    g.ModelSense = gp.GRB.MINIMIZE
    g.Params.MIPGap = gap
    g.update()
    g.optimize()

    y = { k : (v.X if hasattr(v, 'X') else v) for k, v in y.items() }

    # unnormalized objective
    obj = g.getObjective().getValue()

    return { 'y': y, 'obj': obj,
             'done': int(g.status == gp.GRB.OPTIMAL),
             'gap': g.MIPGap,
             'time': g.Runtime }
