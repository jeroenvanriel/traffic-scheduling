from jobshop import read_instance
from pymongo import MongoClient
from datetime import datetime
from itertools import product
from typing import List, Dict, Tuple

# order the operations according to some priority rule
# construct a schedule by dispatching in this order
# 1. with the "no-idle time"-rule
# 2. without the "no-idle time"-rule

def solve(n, m, ptimes: Dict[Tuple, int], order: List[List[int]]):

    # maximum of processing times
    M = max(ptimes.values()) + 1

    # order the operations according to some priority rule
    # operations is identified as tuple (machine, job)
    ordered_operations = []

    # example: SPT priority dispatching

    # eligible operations are determined by keeping track of the latest scheduled operation for each job
    latest_operation = [-1 for _ in range(n)]

    # while not all operations are scheduled
    while not all([o == m - 1 for o in latest_operation]):

        shortest_p, shortest_o = M, (0, 0)
        for j in range(n):
            # the next (unscheduled) operation for job j
            o = latest_operation[j] + 1
            if o >= m:
                # all operations have been scheduled for this job
                continue

            # the machine for this operation
            i = order[j][o]


            # find the smallest eligible operation
            if ptimes[i,j] < shortest_p:
                shortest_p = ptimes[i,j]
                shortest_o = (i, j)

        ordered_operations.append(shortest_o)
        # the next operation of job j becomes eligible
        _, j = shortest_o
        latest_operation[j] += 1

    # given the operations order, construct a schedule (start times) by dispatching
    # 1. with the "no-idle time"-rule
    # 2. without the "no-idle time"-rule
    y = { (i,j): 0 for i, j in product(range(m), range(n)) }

    # keep track of the completion time of last job of each individual machine
    machine_C = [0 for _ in range(m)]
    # keep track of the completion time of last operation of each individual job
    job_C = [0 for _ in range(n)]

    print(f"computed operation order: {ordered_operations}")

    for i, j in ordered_operations:
        y[i,j] = max(machine_C[i], job_C[j])
        # update completion time
        C = y[i,j] + ptimes[i,j]
        machine_C[i] = C
        job_C[j] = C

    return y


def print_solution(y):
    print(10*"-" + "solution" + 10*"-")
    for k, v in y.items():
        print(f"y{k}: {v}")


def save_to_mongodb(n, m, ptimes, y):
    # load network from mongodb
    client = MongoClient("mongodb://127.0.0.1:3001/meteor")
    db = client.meteor

    def one_based(k):
        return tuple(x + 1 for x in k)

    schedule = {
        'date': datetime.now(),
        'n': n,
        'm': m,
        'y': {},
        'ptimes': {},
    }

    for k, v in y.items():
        schedule['y'][str(one_based(k))] = v

    for k, v in ptimes.items():
        schedule['ptimes'][str(one_based(k))] = v

    db.schedules.insert_one(schedule)


if __name__ == "__main__":
    import sys
    file_name = sys.argv[1]
    n, m, ptime, order = read_instance(file_name)
    y = solve(n, m, ptime, order)
    print_solution(y)
    save_to_mongodb(n, m, ptime, y)
