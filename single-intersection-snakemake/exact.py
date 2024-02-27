import gurobipy as gp
import numpy as np
from itertools import product, combinations
from glob import glob
import re, time, os


def check_platoons(releases, lengths):
    """Check whether release and length specify non-overlapping platoons for
    each lane (so overlap may exist between lanes, but not on the same lane)."""

    for release, length in zip(releases, lengths):
        if not (length > 0).all():
            raise Exception("Platoon lengths should be positive.")

        end_times = release + length

        end_times = np.roll(end_times, 1)
        end_times[0] = 0

        if not (release >= end_times).all():
            raise Exception("There are overlapping platoons.")


def solve(switch, release, length, gap=0.0, timelimit=0, consolelog=False, logfile=None):
    """Solve the single intersection problem with complete knowledge of future
    as a MILP.

    Args:

    switch = switch-over time

    release = earliest arrival times, given as list of K arrays of length n_k
    (number of arrivals on lane k)

    length = platoon lengths, list of K arrays of length n_k (number of arrivals
    on lane k)

    Returns: (y, o, obj)

    y = The starting times y[k, j] for each vehicle j on lane k.

    o = The binary
    decisions o[k1, k2, j, l] for each combinations of lanes k1 and k2 with
    vehicle j on lane k1 and vehicle l on lane k2.

    obj = The objective is the sum of (y[k, j] - release[k][j]) * length[k][j]
    over all lanes and corresponding vehicles.
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

    # number of lanes
    K = len(release)

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
    return { k : (v.X if hasattr(v, 'X') else v) for k, v in y.items() }, \
            { k : (v.X if hasattr(v, 'X') else v) for k, v in o.items() }, \
            - (g.getObjective().getValue() - travel_time) # note the minus sign (cost)


if __name__ == "__main__":

    # read parameters
    in_file = snakemake.input[0]

    gap = snakemake.params["gap"]
    timelimit = snakemake.params["timelimit"]
    consolelog = snakemake.params["consolelog"]
    logfile = getattr(snakemake.params, "logfile", None)

    # load instance
    p = np.load(in_file)
    start = time.time()

    K = int(p['K'])
    s = p['s']

    # extract arrivals for each lanes
    release = []
    length = []
    for k in range(K):
        release.append(p[f"arrival{k}"])
        length.append(p[f"length{k}"])

    y, _, obj = solve(s, release, length, gap=gap, timelimit=timelimit, consolelog=consolelog, logfile=logfile)
    print(f"---- exact objective = {obj}")
    wall_time = time.time() - start

    # derive start and end times in standard format
    res = {}
    n = [len(r) for r in release]
    for k in range(K):
        res[f'start_time_{k}'] = np.array([y[k, j] for j in range(n[k])])
        res[f'end_time_{k}'] = np.array([y[k, j] + length[k][j] for j in range(n[k])])

    # write results
    out_file = snakemake.output[0]
    np.savez(out_file, **res, K=K, s=s, obj=obj, time=wall_time, gap=gap)
