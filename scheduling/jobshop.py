import gurobipy as gp
from itertools import product, combinations

def read_instance(file):
    with open(file, 'r') as f:
        n, m = map(int, f.readline().split())

        ptime = [[] for _ in range(n)]
        order = [[] for _ in range(n)]

        # Read operation times for each job.
        for j in range(n):
            ptime[j] = list(map(int, f.readline().split()))

        # Read machine order for each job.
        for j in range(n):
            order[j] = list(map(int, f.readline().split()))

    # Verify whether all processing times are provided.
    # N.B. When a job does not have an operation for a particular machine, the
    # corresponding processing time should just be set to zero.
    for pt in ptime:
        assert len(pt) == m

    # Verify whether order is a valid permutation of the machines.
    for o in order:
        assert len(o) == m
        assert set(o) == set(range(1, m + 1))

    # Transpose the list of processing times, to align with the conventional
    # p_{ij} notation, where i=machine, j=job.
    ptime = list(map(list, zip(*ptime)))

    # Make machine numbers start at zero.
    order = list(map(lambda l: list(map(lambda x: x-1, l)), order))

    return n, m, ptime, order


def print_solution(y, s):
    print(10*"-" + "solution (rounded)" + 10*"-")
    for k, v in y.items():
        print(f"y{k}: {round(v.X, 4)}")

    for k, v in s.items():
        print(f"s{k}: {v.X}")


def solve(n, m, p, o):
    g = gp.Model()

    # big-M
    M = 1000

    # non-negative makespan
    makespan = g.addVar(obj=1, vtype=gp.GRB.CONTINUOUS, name="makespan")

    # non-negative starting times
    y = { (i, j):
            g.addVar(obj=0, vtype=gp.GRB.CONTINUOUS, name=f"y_{i}_{j}")
            for i, j in product(range(m), range(n)) }

    # disjunctive variables for "selecting" disjunctive arcs
    # (machine i, job j, job l)
    # true = job l before job j
    s = { (i, j, l):
            g.addVar(obj=0, vtype=gp.GRB.BINARY, name=f"s_{i}_{j}_{l}")
            for j, l in combinations(range(n), 2)
            for i in range(m) }

    # order operations for each job
    for j in range(n):
        for i, k in zip(o[j][:-1], o[j][1:]):
            g.addConstr(y[i, j] + p[i][j] <= y[k, j])

    # makespan constraints
    for i, j in product(range(m), range(n)):
        g.addConstr(y[i, j] + p[i][j] <= makespan)

    # order operations on each machine
    for i in range(m):
        for j, l in combinations(range(n), 2):
            # j before l
            g.addConstr(y[i,j] + p[i][j] <= y[i,l] + s[i,j,l] * M)

            # l before j
            g.addConstr(y[i,l] + p[i][l] <= y[i,j] + (1 - s[i,j,l]) * M)

    g.ModelSense = gp.GRB.MINIMIZE
    g.update()
    g.optimize()

    print_solution(y,s)


if __name__ == "__main__":
    import sys
    file_name = sys.argv[1]
    n, m, ptime, order = read_instance(file_name)
    solve(n, m, ptime, order)
