import gurobipy as gp
import numpy as np
from itertools import product, combinations
import os


def solve(instance, gap=0.0, timelimit=0, consolelog=False, logfile=None):
    """Solve a single intersection scheduling problem as a MILP."""

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

    release = instance['release']
    length = instance['length']
    switch = instance['switch']  # switch-over time

    N = len(release) # number of lanes
    n = [len(r) for r in release] # number of arrivals per lane

    # big-M
    M = 1000

    ### Variables

    # non-negative starting times
    y = {}
    for k in range(N):
        for j in range(n[k]):
            y[k, j] = g.addVar(obj=1, vtype=gp.GRB.CONTINUOUS, name=f"y_{k}_{j}")
            g.addConstr(y[k, j] >= release[k][j])

    ### Constraints

    # conjunctions
    for k in range(N):
        for j in range(n[k] - 1):
            g.addConstr(y[k, j] + length[k][j] <= y[k, j + 1])

    # disjunctions
    for k1, k2 in combinations(range(N), 2):
        for j, l in product(range(n[k1]), range(n[k2])):
            oc = g.addVar(obj=0, vtype=gp.GRB.BINARY, name=f"o_{k1}_{k2}_{j}_{l}")

            g.addConstr(y[k1, j] + length[k1][j] + switch <= y[k2, l] + oc * M)
            g.addConstr(y[k2, l] + length[k2][l] + switch <= y[k1, j] + (1 - oc) * M)

    ### Solving

    g.ModelSense = gp.GRB.MINIMIZE
    g.Params.MIPGap = gap
    g.update()
    g.optimize()

    res = { 'y': [], 'obj': g.getObjective().getValue() }
    y = { k : (v.X if hasattr(v, 'X') else v) for k, v in y.items() }
    for k in range(N):
        res['y'].append(np.array([y[k, j] for j in range(n[k])]))

    return res
