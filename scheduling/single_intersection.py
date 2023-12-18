import gurobipy as gp
import numpy as np
from itertools import product
from pymongo import MongoClient
from datetime import datetime
import sys

##
# Currently, k=2 lanes is hardcoded. This can be easily generalized, where the
# most important change would be related to the disjunctions.
#
# Currently, each lane has precisely n arrivals.
#
# The objective returned here is \sum_{i} d_i = \sum_{i} (y_i - r_i) * l_i
# where r_i, l_i are release date and length of platoon i, respectively.
##

def read_instance(file):
    with open(file, 'r') as f:
        def readline():
            l = f.readline().strip()
            if l and len(l) > 0 and l[0] != '#':
                return l.split()
            else:
                return readline()

        n, = map(int, readline())
        switch, = map(float, readline())

        release = np.empty((2, n))
        length = np.empty((2, n))

        # each lane has n arrivals
        for j in range(2 * n):
            line = readline()
            l = j // n

            # Read release date for each job.
            release[l, j % n] = float(line[0])

            # Read machine order for each job.
            length[l, j % n] = float(line[1])

    check_platoons(release, length)

    return n, switch, release, length


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


def solve(n, switch, release, length, gap=0, log=True):
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
           { k : (v.X if hasattr(v, 'X') else v) for k, v in o.items() }, \
            g.getObjective().getValue() - (release * length).sum()


def print_solution(y, o, obj):
    print(10*"-" + "solution (rounded)" + 10*"-")
    print(f"total completion time: {obj}")

    def one_based(k):
        return [x + 1 for x in k]

    for k, v in y.items():
        print(f"y{one_based(k)}: {round(v, 4)}")

    for k, v in o.items():
        print(f"o{one_based(k)}: {v}")


def save_to_mongodb(n, y, switch, release, length):
    client = MongoClient("mongodb://127.0.0.1:3001/meteor")
    db = client.meteor

    schedule = {
        'date': datetime.now(),
        'switch': switch,
        'y': {},
        'ptimes': {},
        'release': {},
        'colors': {},
    }

    for l in range(2):
        for i in range(n):
            key = (1, l * n + i)

            schedule['y'][str(key)] = float(y[l, i])
            schedule['release'][str(key)] = float(release[l][i])
            schedule['ptimes'][str(key)] = float(length[l][i])
            schedule['colors'][str(key)] = 'red' if l == 0 else 'blue'

    db.schedules.insert_one(schedule)


if __name__ == "__main__":
    instance = sys.argv[1]
    n, switch, release, length = read_instance(instance)

    y, o, obj = solve(n, switch, release, length)

    print_solution(y, o, obj)

    save_to_mongodb(n, y, switch, release, length)
