import gurobipy as gp
import numpy as np
from itertools import product, combinations
import os


def solve(instance, gap=0.0, timelimit=0, consolelog=False, logfile=None, cutting_planes=None):
    """Solve a single intersection scheduling problem as a MILP.

    `cutting_planes` is a list/set specifying which cutting planes to add,
    possible choices are integers:
        1 - transitive cutting planes
        2 - necessary conjunctive cutting planes
        3 - necessary disjunctive cutting planes
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

    if cutting_planes is None:
        cutting_planes = []

    release = instance['release'] # earliest crossing times (a_i)
    length = instance['length']   # vehicle follow time (rho)
    switch = instance['switch']   # switch-over time (sigma)

    R = len(release) # number of routes
    n = [len(r) for r in release] # number of arrivals per route

    # big-M
    M = 1000

    ### Variables

    # non-negative starting times
    y = {}
    for r in range(R):
        for k in range(n[r]):
            y[r, k] = g.addVar(obj=1, vtype=gp.GRB.CONTINUOUS, name=f"y_{r}_{k}")
            g.addConstr(y[r, k] >= release[r][k])

    ### Constraints

    # indicator variables for the necessary cutting planes
    delta = {}

    # conjunctions
    for r in range(R):
        for k in range(n[r] - 1):
            g.addConstr(y[r, k] + length[r][k] <= y[r, k + 1])

            # definition of delta
            if 2 in cutting_planes or 3 in cutting_planes:
                delta[r, k] = g.addVar(obj=0, vtype=gp.GRB.BINARY, name=f"delta_{r}_{k}")
                g.addConstr(y[r, k] + length[r][k] <= release[r][k + 1] + delta[r, k] * M)
                g.addConstr(y[r, k] + length[r][k] >= release[r][k + 1] - (1 - delta[r, k]) * M)

            if 2 in cutting_planes:
                g.addConstr(y[r, k] + length[r][k] >= y[r, k + 1] - (1 - delta[r, k]) * M)


    # disjunctions
    o = {}
    for r1, r2 in combinations(range(R), 2):
        for k, l in product(range(n[r1]), range(n[r2])):
            oc = g.addVar(obj=0, vtype=gp.GRB.BINARY, name=f"o_{r1}_{k}_{r2}_{l}")
            o[r1, k, r2, l] = oc

            g.addConstr(y[r1, k] + length[r1][k] + switch <= y[r2, l] + oc * M)
            g.addConstr(y[r2, l] + length[r2][l] + switch <= y[r1, k] + (1 - oc) * M)

            # transitive cutting planes
            if 1 in cutting_planes:
                g.addConstr(gp.quicksum(o[r1, p, r2, q] for p in range(0, k) for q in range(l + 1, n[r2])) <= o[r1, k, r2, l] * M)


    # necessary disjunctive cutting planes
    if 3 in cutting_planes:
        for r1, r2 in combinations(range(R), 2):
            for l in range(n[r1] - 1): # for all conjunctive pairs (i, j) = ((r1, l), (r1, l + 1))
                for k in range(n[r2]):
                    g.addConstr(delta[r1, l] + (1 - o[r1, l, r2, k]) + o[r1, l + 1, r2, k] <= 2)
                    g.addConstr(delta[r1, l] + o[r1, l, r2, k] + (1 - o[r1, l + 1, r2, k]) <= 2)

    ### Solving

    g.ModelSense = gp.GRB.MINIMIZE
    g.Params.MIPGap = gap
    g.update()
    g.optimize()

    res = { 'y': [], 'obj': g.getObjective().getValue() }
    y = { r : (v.X if hasattr(v, 'X') else v) for r, v in y.items() }
    for r in range(R):
        res['y'].append(np.array([y[r, j] for j in range(n[r])]))

    res['done'] = int(g.status == gp.GRB.OPTIMAL)
    res['gap'] = g.MIPGap
    res['time'] = g.Runtime

    return res
