import gurobipy as gp
import numpy as np
from itertools import product, combinations
import os


def solve(instance, gap=0.0, timelimit=0, consolelog=False, logfile=None, cutting_planes=0):
    """Solve a single intersection scheduling problem as a MILP.

    cutting_planes:
    0 - no
    1 - type 1
    2 - type 2
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
    beta = {}
    for k in range(N):
        for j in range(n[k]):
            y[k, j] = g.addVar(obj=1, vtype=gp.GRB.CONTINUOUS, name=f"y_{k}_{j}")
            g.addConstr(y[k, j] >= release[k][j])

    ### Constraints

    # conjunctions
    for k in range(N):
        for j in range(n[k] - 1):
            g.addConstr(y[k, j] + length[k][j] <= y[k, j + 1])

            # exhaustive rule constraints
            if cutting_planes > 0:
                beta[k, j] = g.addVar(obj=0, vtype=gp.GRB.BINARY, name=f"beta_{k}_{j}")

                g.addConstr(y[k, j] + length[k][j] <= release[k][j + 1] + beta[k, j] * M)
                g.addConstr(y[k, j] + length[k][j] >= release[k][j + 1] - (1 - beta[k, j]) * M)

                g.addConstr(y[k, j] + length[k][j] >= y[k, j + 1] - (1 - beta[k, j]) * M)


    # disjunctions
    o = {}
    for k1, k2 in combinations(range(N), 2):
        for j, l in product(range(n[k1]), range(n[k2])):
            oc = g.addVar(obj=0, vtype=gp.GRB.BINARY, name=f"o_{k1}_{j}_{k2}_{l}")
            o[k1, j, k2, l] = oc

            g.addConstr(y[k1, j] + length[k1][j] + switch <= y[k2, l] + oc * M)
            g.addConstr(y[k2, l] + length[k2][l] + switch <= y[k1, j] + (1 - oc) * M)

    if cutting_planes == 2:
        for k, k2 in combinations(range(N), 2):
            for j in range(n[k] - 1): # for all conjunctive pairs (j, j + 1)
                for j2 in range(n[k2]):
                    g.addConstr(beta[k, j] + (1 - o[k, j, k2, j2]) + o[k, j + 1, k2, j2] <= 2)
                    g.addConstr(beta[k, j] + o[k, j, k2, j2] + (1 - o[k, j + 1, k2, j2]) <= 2)


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
