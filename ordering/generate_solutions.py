import time
import glob, re
import pickle, csv
import numpy as np
from tqdm import trange

from exact import solve
from util import lane_order

global_start = time.time()

times_file = open('../report/data/running_times.csv', 'w', newline='')
writer = csv.writer(times_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
writer.writerow(["set_id", "instance_id", "cut0", "cut1", "cut2"])


# load all the instances_{i}.pkl files in the current directory
files = glob.glob("data/instances_*.pkl")
for file in files:
    print(f"processing {file}")
    i = int(re.findall(r'\d+', file)[0])
    with open(file, 'rb') as file:
        instances = pickle.load(file)

    schedules = [[], [], []]
    times = np.empty((len(instances), 3))

    def solve_all(c): # with timing
        for j in trange(len(instances)):
            start = time.time()
            schedules[c].append(solve(instances[j], cutting_planes=c))
            times[j][c] = time.time() - start

    solve_all(0) # no cutting planes
    solve_all(1) # cutting planes type 1
    solve_all(2) # cutting planes type 2

    # report times for this set
    for j, row in enumerate(times):
        # i = data set id
        # j = instance id
        # row contains 3 running times; one for each type of MILP
        writer.writerow([i, j, *row.tolist()])

    # we can optinally verify the cutting planes by comparing solutions

    # store schedules with their lane order
    with open(f"data/schedules_{i}.pkl", 'wb') as out_file:
        pickle.dump([(schedule, lane_order(schedule)) for schedule in schedules[0]], out_file)

times_file.close()

print(f"total time: {time.time() - global_start}")
