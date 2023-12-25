import gurobipy as gp
import numpy as np
from itertools import product


def solve(locations, delta, p, switch, release, lane, gap=0.0, log=True):
    """
    locations = number of locations including intersection, excluding entrypoint
    delta = delta t, time to travel to next location at cruise speed
    p = time between consecutive arrivals from same lane
    switch = switch-over time between vehicles of distinct lanes
    release[j] = arrival time of vehicle j (TODO: now assuming sorted)
    lane[j] = lane number of vehicle j (TODO: assume increasing sequence)
    """

    env = gp.Env(empty=True)
    if not log:
        # disable console logging and license information
        env.setParam('OutputFlag', 0)

    env.start()
    g = gp.Model(env=env)

    # big-M
    M = 1000

    n = len(release)

    ### Variables

    # non-negative departure times
    y = {}
    for i in range(locations + 1):
        for j in range(n): # vehicles
            # TODO: select only departure time at subset of locations (e.g., all
            # intersections, or "last on route" intersections)
            y[i, j] = g.addVar(obj=1, vtype=gp.GRB.CONTINUOUS, name=f"y_{i}_{j}")

    o = {}

    ### Constraints

    # per vehicle
    for j in range(n):
        for i in range(locations + 1):
            if i == 0:
                # release date on first location
                g.addConstr(y[i, j] >= release[j])
            else:
                # departure time after arrival time
                g.addConstr(y[i-1, j] + delta <= y[i, j])

    # vehicles per group
    lane_groups = np.split(np.arange(n), np.unique(lane, return_index=True)[1][1:])

    # conjunctions
    for group in lane_groups:
        for j in group[0:-1]:
            for i in range(1, locations + 1):
                g.addConstr(y[i, j] + p <= y[i-1, j+1] + delta)

    # disjunctions
    # TODO: assuming two lanes for now
    for j, l in product(lane_groups[0], lane_groups[1]):
        o[j, l] = g.addVar(obj=0, vtype=gp.GRB.BINARY, name=f"o_{j}_{l}")

        # intersection is the last location
        i = locations

        g.addConstr(y[i, j] + p + switch <= y[i-1, l] + o[j, l] * M)
        g.addConstr(y[i, l] + p + switch <= y[i-1, j] + (1 - o[j, l]) * M)

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
