import gurobipy as gp
from itertools import product, combinations
from pymongo import MongoClient
from datetime import datetime

def read_instance(file):
    with open(file, 'r') as f:
        n, m = map(int, f.readline().split())
        ptime, switch = map(int, f.readline().split())

        # Read adjacency matrix for machine distance graph.
        distance = [[] for _ in range(m)]
        for i in range(m):
            distance[i] = list(map(int, f.readline().split()))

        order = [[] for _ in range(n)]
        release = [0 for _ in range(n)]

        for j in range(n):
            numbers = list(map(int, f.readline().split()))

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

    # Make sure the adjacency matrix is symmetric. We only expect the lower
    # triangular part and use it to fill the upper part.
    for i in range(m):
        for k in range(i+1, m):
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


def solve(n, m, ptime, switch, release, order, distance):
    g = gp.Model()

    # big-M
    M = 100

    # non-negative makespan
    makespan = g.addVar(obj=1, vtype=gp.GRB.CONTINUOUS, name="makespan")

    # non-negative starting times
    y = {}
    for j in range(n):
        # Initialize the starting times at incoming edges based on release date.
        y[(order[j][0], j)] = release[j]

        # We add variables only for times that are actually relevant.
        for i in order[j][1:]:
            y[(i, j)] = g.addVar(obj=1, vtype=gp.GRB.CONTINUOUS, name=f"y_{i}_{j}")

    # Determine pairs of vehicles that conflict at a shared intersection and
    # create disjunctive variables for "selecting" disjunctive arcs
    #   (machine i, job j, job l)
    #   true = job l before job j
    conflicts = []
    s = {}
    for j, l in combinations(range(n), 2):
        common_intersections = set(order[j]).intersection(set(order[l]))

        for i in common_intersections:
            if order[j].index(i) == 0 or order[l].index(i) == 0:
                # Overlap in entrypoint can never be a conflict.
                continue

            if order[j][order[j].index(i) - 1] != order[l][order[l].index(i) - 1]:
                # Conflict found.
                conflicts.append((i, j, l))
                s[(i, j, l)] = g.addVar(obj=0, vtype=gp.GRB.BINARY, name=f"s_{i}_{j}_{l}")

    # Enforce conflicts at intersections (disjunctions).
    for i, j, l in conflicts:
        # j before l
        g.addConstr(y[i,j] + ptime + switch <= y[i,l] + s[i,j,l] * M)

        # l before j
        g.addConstr(y[i,l] + ptime + switch <= y[i,j] + (1 - s[i,j,l]) * M)

    # Fix order of vehicles on the same incoming lane based on release dates.
    for i in range(m):
        # Collect all vehicles that start at entrypoint i.
        vehicles = []
        for j in range(n):
            if order[j][0] == i:
                vehicles.append((j, release[j]))

        if len(vehicles) <= 1:
            continue

        # Get the intersection (second machine) to which all these vehicles are
        # travelling.
        k = order[vehicles[0][0]][1]

        # Order these vehicles based on release date...
        lane_order = list(map(lambda x: x[0], sorted(vehicles, key=lambda x: x[1])))

        # ...and add constraints for each consecutive pair in the order.
        for j, l in zip(lane_order[:-1], lane_order[1:]):
            g.addConstr(y[k, j] + ptime <= y[k, l])

    # Enforce machine order (operations order per job) and travel time.
    #
    # N.B.: We could assume that the entrypoint does not require processing
    # time, but that would complicate the implementation. Essentially, it does
    # not make a difference, as is equivalent to a uniform shift of the release
    # dates of all jobs.
    for j in range(n):
        for i, k in zip(order[j][0:-1], order[j][1:]):
            g.addConstr(y[i, j] + ptime + distance[i][k] <= y[k, j])

    # makespan constraints
    for i, j in y.keys():
        g.addConstr(y[i, j] + ptime <= makespan)


    g.ModelSense = gp.GRB.MINIMIZE
    g.update()
    g.optimize()

    # Somehow, we need to do this inside the current function definition,
    # otherwise the Gurobi variables don't expose the .X attribute anymore.
    return { k : (v.X if hasattr(v, 'X') else v) for k, v in y.items() }, \
           { k : v.X for k, v in s.items() }


def save_to_mongodb(y, s, ptime, switch):
    # load network from mongodb
    client = MongoClient("mongodb://127.0.0.1:3001/meteor")
    db = client.meteor

    schedule = {
        'date': datetime.now(),
        'y': {},
        's': {},
        'ptime': ptime,
        'swtich': switch,
    }

    def one_based(k):
        return tuple(x + 1 for x in k)

    for k, v in y.items():
        schedule['y'][str(one_based(k))] = v

    for k, v in s.items():
        schedule['s'][str(one_based(k))] = v

    db.schedules.insert_one(schedule)


if __name__ == "__main__":
    n, m, ptime, switch, release, order, distance = read_instance("traffic1.txt")
    y, s = solve(n, m, ptime, switch, release, order, distance)

    print_solution(y, s)

    save_to_mongodb(y, s, ptime, switch)
