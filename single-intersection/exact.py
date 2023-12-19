import gurobipy as gp
import numpy as np
from itertools import product
from glob import glob
import re, time

##
# Currently, k=2 lanes is hardcoded. This can be easily generalized, where the
# most important change would be related to the disjunctions.
#
# Currently, each lane has precisely n arrivals.
#
# The objective returned here is \sum_{i} d_i = \sum_{i} (y_i - r_i) * l_i
# where r_i, l_i are release date and length of platoon i, respectively.
##


def check_platoons(release, length):
    """Check whether release and length specify non-overlapping platoons for
    each lane (so overlap may exist between lanes, but not on the same lane)."""

    if not (length > 0).all():
        raise Exception("Platoon lengths should be positive.")

    end_times = release + length

    end_times = np.roll(end_times, 1, axis=1)
    end_times[:,0] = 0

    if not (release >= end_times).all():
        raise Exception("There are overlapping platoons.")


def solve(n, switch, release, length, gap=0.0, log=True):
    env = gp.Env(empty=True)
    if not log:
        # disable console logging and license information
        env.setParam('OutputFlag', 0)

    env.start()
    g = gp.Model(env=env)

    # big-M
    M = 1000

    ### Variables

    # non-negative starting times
    y = {}
    for k in range(2):
        for j in range(n):
            y[k, j] = g.addVar(obj=length[k, j], vtype=gp.GRB.CONTINUOUS, name=f"y_{k}_{j}")
            g.addConstr(y[k, j] >= release[k, j])

    o = {}

    ### Constraints

    # conjunctions
    for k in range(2):
        for j in range(n - 1):
            g.addConstr(y[k, j] + length[k, j] <= y[k, j + 1])

    # disjunctions
    for j, l in product(range(n), range(n)):
        o[j, l] = g.addVar(obj=0, vtype=gp.GRB.BINARY, name=f"o_{j}_{l}")

        g.addConstr(y[0, j] + length[0, j] + switch <= y[1, l] + o[j, l] * M)
        g.addConstr(y[1, l] + length[1, l] + switch <= y[0, j] + (1 - o[j, l]) * M)

    ### Solving

    g.ModelSense = gp.GRB.MINIMIZE
    g.Params.MIPGap = gap
    g.update()
    g.optimize()

    # Somehow, we need to do this inside the current function definition,
    # otherwise the Gurobi variables don't expose the .X attribute anymore.
    return { k : (v.X if hasattr(v, 'X') else v) for k, v in y.items() }, \
            - (g.getObjective().getValue() - (release * length).sum())


if __name__ == "__main__":

    log = True
    gap = 0.1 # optimality gap

    # solve all the instances
    for in_file in glob("./instances/*.npz"):
        m = re.match(r".\/instances\/instance\_(\d+)\_(\d+)\.npz", in_file)
        if m is None:
           raise Exception("Incorrect instance file name.")
        ix = m.group(1) # experiment number
        i = m.group(2)  # sample number

        p = np.load(in_file)
        start = time.time()
        y, obj = solve(p['n'], p['s'], p['arrival'], p['length'], log=log, gap=gap)
        wall_time = time.time() - start

        out_file = f"./schedules/exact_{ix}_{i}.npz"
        np.savez(out_file, y=y, obj=obj, time=wall_time, gap=gap)
