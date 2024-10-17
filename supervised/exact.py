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

    switch = instance['s']  # switch-over time
    K = instance['K']       # number of lanes

    # extract arrivals for each lanes
    release = []
    length = []
    for k in range(K):
        release.append(instance[f"arrival{k}"])
        length.append(instance[f"length{k}"])

    assert K == len(release)

    # number of arrivals per lane
    n = [len(r) for r in release]

    # big-M
    M = 1000

    ### Variables

    # non-negative starting times
    y = {}
    for k in range(K):
        for j in range(n[k]):
            y[k, j] = g.addVar(obj=length[k][j], vtype=gp.GRB.CONTINUOUS, name=f"y_{k}_{j}")
            g.addConstr(y[k, j] >= release[k][j])

    ### Constraints

    # conjunctions
    for k in range(K):
        for j in range(n[k] - 1):
            g.addConstr(y[k, j] + length[k][j] <= y[k, j + 1])

    # disjunctions
    o = {}
    for k1, k2 in combinations(range(K), 2):
        for j, l in product(range(n[k1]), range(n[k2])):
            oc = g.addVar(obj=0, vtype=gp.GRB.BINARY, name=f"o_{k1}_{k2}_{j}_{l}")
            o[k1, k2, j, l] = oc

            g.addConstr(y[k1, j] + length[k1][j] + switch <= y[k2, l] + oc * M)
            g.addConstr(y[k2, l] + length[k2][l] + switch <= y[k1, j] + (1 - oc) * M)

    ### Solving

    g.ModelSense = gp.GRB.MINIMIZE
    g.Params.MIPGap = gap
    g.update()
    g.optimize()

    # total travel time that is always induced, so not part of objective
    travel_time = 0
    for k in range(K):
        travel_time += (release[k] * length[k]).sum()

    # Somehow, we need to do this inside the current function definition,
    # otherwise the Gurobi variables don't expose the .X attribute anymore.
    y = { k : (v.X if hasattr(v, 'X') else v) for k, v in y.items() }
    obj = - (g.getObjective().getValue() - travel_time) # note the minus sign (cost)

    # derive start and end times in standard format
    res = { **instance, 'obj': obj }
    n = [len(r) for r in release]
    for k in range(K):
        res[f'start_time_{k}'] = np.array([y[k, j] for j in range(n[k])])
        res[f'end_time_{k}'] = np.array([y[k, j] + length[k][j] for j in range(n[k])])

    return res
