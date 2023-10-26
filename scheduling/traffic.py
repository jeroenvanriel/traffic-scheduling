import gurobipy as gp
from itertools import product, combinations
from pymongo import MongoClient
from datetime import datetime
import sys

def read_instance(file):
    with open(file, 'r') as f:
        def readline():
            l = f.readline().strip()
            if l and len(l) > 0 and l[0] != '#':
                return l.split()
            else:
                return readline()

        n, m = map(int, readline())
        ptime, switch = map(int, readline())

        # Read adjacency matrix for machine distance graph.
        distance = [[] for _ in range(m)]
        for i in range(m):
            distance[i] = list(map(int, readline()))

        order = [[] for _ in range(n)]
        release = [0 for _ in range(n)]

        for j in range(n):
            numbers = list(map(int, readline()))

            # Read release date for each job.
            release[j] = numbers[0]

            # Read machine order for each job.
            order[j] = numbers[1:]

    # Verify whether order is a valid permutation of a subset of machines.
    for o in order:
        assert len(o) <= m
        assert set(o) <= set(range(1, m + 1))
        assert len([n for n in o if o.count(n) > 1]) == 0 # no duplicates

    # Make machine numbers start at zero.
    order = list(map(lambda l: list(map(lambda x: x-1, l)), order))

    # If entries in the upper triangular part are missing, we use the lower
    # triangular part to fill the upper part.
    for i in range(m):
        for k in range(i+1, m):
            if len(distance[i]) > k:
                distance[i][k] = distance[k][i]
            else:
                distance[i].append(distance[k][i])

    return n, m, ptime, switch, release, order, distance


def print_solution(y, s):
    print(10*"-" + "solution (rounded)" + 10*"-")

    def one_based(k):
        return [x + 1 for x in k]

    for k, v in y.items():
        print(f"y{one_based(k)}: {round(v, 4)}")

    for k, v in s.items():
        print(f"s{one_based(k)}: {v}")


def common_substring(a, b):
    """Assumption: at most one common substring."""
    res = []
    first = None

    for x in a:
        if x in b:
            # found first match
            first = x
            break

    if first is not None:
        # match elements from first
        for x, y in zip(a[a.index(first):], b[b.index(first):]):
            if x == y:
                res.append(x)
            else:
                break

    return res


def solve(n, m, ptime, switch, release, order, distance, entrypoints, exitpoints):
    g = gp.Model()

    # big-M
    M = 100


    ### Variables

    # non-negative starting times
    y = {}
    for j in range(n):
        # Initialize the starting times at incoming edges based on release date.
        y[(order[j][0], j)] = release[j]

        # We add variables only for times that are actually relevant (so only
        # for the machines/intersection that the vehicle actually visits).
        for ix in range(1, len(order[j])):
            # Add the start time of the last operation to the objective.
            #
            # Note that the total completion time is the start time at the last
            # intersection, which must be an exitpoint and does not
            # have any processing time.
            obj = int(ix == len(order[j]) - 1)
            i = order[j][ix]
            y[(i, j)] = g.addVar(obj=obj, vtype=gp.GRB.CONTINUOUS, name=f"y_{i}_{j}")

    s = {}

    ### For all pairs of vehicles, add constraints to enforce order, either based on
    #   - initial ordering at entrypoint, defined by release dates, or
    #   - conflicts at intersections.
    for j, l in combinations(range(n), 2):
        p = common_substring(order[j], order[l])

        if len(p) == 0:
            continue

        # Handle substring p.
        # Assumption: len(p) > 0
        #
        # It should be possible to later reuse the following code when we drop the
        # DAG assumption. In that case, each overlapping substring should be handled
        # by the following procedure.

        # merge point and tail (may be empty)
        mp, *tail = p

        if mp in exitpoints:
            raise Exception("Exit point may not have more than one incoming edge.")

        if mp in entrypoints:
            # Base the order on the release dates.
            s[(mp, j, l)] = (s0 := (0 if release[j] <= release[l] else 1))

            if s0 == 0:
                first, second = j, l
            else:
                first, second = l, j

            # Keep this order in the tail.
            for k in tail:
                g.addConstr(y[k, first] + ptime <= y[k, second])
        else:
            # Merge point must be an intersection, hence we are dealing with a
            # conflict.
            s[(mp, j, l)] = (s0 := g.addVar(obj=0, vtype=gp.GRB.BINARY, name=f"s_{mp}_{j}_{l}"))

            g.addConstr(y[mp, j] + ptime + switch <= y[mp, l] + s0 * M)
            g.addConstr(y[mp, l] + ptime + switch <= y[mp, j] + (1 - s0) * M)

            # Enforce the same order in the tail, but without switch-over time.
            for k in tail:
                g.addConstr(y[mp, j] + ptime <= y[mp, l] + s0 * M)
                g.addConstr(y[mp, l] + ptime <= y[mp, j] + (1 - s0) * M)


    ### For each vehicle: enforce machine order (among operations) and travel time.
    # N.B.: We assume that entrypoints (and exitpoints, but that is not relevant
    # for these constraints) do not require processing time.
    for j in range(n):
        for i, k in zip(order[j][0:-1], order[j][1:]):
            g.addConstr(y[i, j] \
                        + (ptime if i not in entrypoints else 0) \
                        + distance[i][k] <= y[k, j])


    g.ModelSense = gp.GRB.MINIMIZE
    g.update()
    g.optimize()

    # Somehow, we need to do this inside the current function definition,
    # otherwise the Gurobi variables don't expose the .X attribute anymore.
    return { k : (v.X if hasattr(v, 'X') else v) for k, v in y.items() }, \
           { k : (v.X if hasattr(v, 'X') else v) for k, v in s.items() }


def save_to_mongodb(y, s, ptime, switch, entrypoints, exitpoints):
    # load network from mongodb
    client = MongoClient("mongodb://127.0.0.1:3001/meteor")
    db = client.meteor

    def one_based(k):
        return tuple(x + 1 for x in k)

    schedule = {
        'date': datetime.now(),
        'y': {},
        's': {},
        'ptime': ptime,
        'swtich': switch,
        'entrypoints': list(one_based(entrypoints)),
        'exitpoints': list(one_based(exitpoints)),
    }

    for k, v in y.items():
        schedule['y'][str(one_based(k))] = v

    for k, v in s.items():
        schedule['s'][str(one_based(k))] = v

    db.schedules.insert_one(schedule)


if __name__ == "__main__":
    instance = sys.argv[1]
    n, m, ptime, switch, release, order, distance = read_instance(instance)

    # Collect the entrypoints (first machine in machine order list) of all jobs.
    entrypoints = {order[j][0] for j in range(n)}
    # Collect the exitpoints (last machine in machine order list) of all jobs.
    exitpoints = {order[j][-1] for j in range(n)}

    y, s = solve(n, m, ptime, switch, release, order, distance, entrypoints, exitpoints)

    print_solution(y, s)

    save_to_mongodb(y, s, ptime, switch, list(entrypoints), list(exitpoints))
