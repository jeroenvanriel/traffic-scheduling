import gurobipy as gp
import numpy as np
from itertools import product, combinations
from glob import glob
import re, time, os

##
# Currently, K=2 lanes is hardcoded. This can be easily generalized, where the
# most important change would be related to the disjunctions.
#
# Currently, each lane has precisely n arrivals.
#
# The objective returned here is \sum_{i} d_i = \sum_{i} (y_i - r_i) * l_i
# where r_i, l_i are release date and length of platoon i, respectively.
##


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


def solve(switch, release, length, gap=0.0, log=True):
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

    obj = The objective is the sum of y[k, j] - release[k][j] over all lanes and
    corresponding vehicles.
    """

    env = gp.Env(empty=True)
    if not log:
        # disable console logging and license information
        env.setParam('OutputFlag', 0)

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
            oc = g.addVar(obj=0, vtype=gp.GRB.BINARY, name=f"o_{j}_{l}")
            o[k1, k2, j, l] = oc

            g.addConstr(y[0, j] + length[0][j] + switch <= y[1, l] + oc * M)
            g.addConstr(y[1, l] + length[1][l] + switch <= y[0, j] + (1 - oc) * M)

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
            g.getObjective().getValue() - travel_time


if __name__ == "__main__":

    log = True
    gap = 0.1 # optimality gap

    # write the schedules here
    os.makedirs(os.path.dirname("./schedules/"), exist_ok=True)

    # solve all the instances
    for in_file in glob("./instances/*.npz"):
        m = re.match(r".\/instances\/instance\_(\d+)\_(\d+)\.npz", in_file)
        if m is None:
           raise Exception("Incorrect instance file name.")
        ix = m.group(1) # experiment number
        i = m.group(2)  # sample number

        p = np.load(in_file)
        start = time.time()

        # extract arrivals for each lanes
        arrivals = []
        lengths = []
        for k in range(p['K']):
            arrivals.append(p[f"arrival{k}"])
            lengths.append(p[f"length{k}"])

        y, _, obj = solve(p['s'], arrivals, lengths, log=log, gap=gap)
        wall_time = time.time() - start

        out_file = f"./schedules/exact_{ix}_{i}.npz"
        np.savez(out_file, y=y, obj=obj, time=wall_time, gap=gap)
