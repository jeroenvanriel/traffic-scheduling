import numpy as np

infile = snakemake.input[0]

instance = np.load(infile)
K = int(instance['K'])
switch_over = float(instance['s'])

### schedule by applying exhaustive service and no-wait switching policy

# we maintain a minheap of events
# event = (time, value, lane, id)
# where we use the following simple convention
# arrival: value == platoon length
# completion: value == -1
# this ensures that completion events are always processed before arrivals
h = []
from heapq import heappush, heappop

# populate the heap with arrival events
for k in range(K):
    arrival = instance[f"arrival{k}"]
    length = instance[f"length{k}"]
    for i, (a, p) in enumerate(zip(arrival, length)):
        event = (a, p, k, i)
        heappush(h, event)

# each entry in queue is (id, platoon length)
from collections import deque
queue = [deque() for k in range(K)]

# current lane
lane = 0

# processing a platoon or idle
busy = False

# produced schedule, list of starting times for each platoon for each lane
schedule = [[] for k in range(K)]

# keep track of completion time of last completed platoon
last_completion_time = 0

# discrete event simulation
try:
    while event := heappop(h):
        (t, p, k, i) = event

        # process arrival
        if p > 0:
            if not busy:
                s = max(t, last_completion_time + (0 if lane == k else switch_over))
                schedule[k].append(s) # record starting time
                C = s + p # completion time
                # add completion time event
                last_completion_time = C
                e = (C, -1, k, i)
                heappush(h, e)
                busy = True
                lane = k

            else:
                # add to queue
                queue[k].append((i, p))


        # process completion
        if p == -1:
            # invariant: busy == True, lane == k
            # process next platoon if any is waiting
            busy = False # or become idle when no more platoons
            # start looking from current lane onwards
            # so in lanes (lane, lane + 1 mod K, lane + 2 mod K, ..., lane - 1 mod K)
            for k in np.roll(np.arange(K), -lane):
                if len(queue[k]) > 0:
                    # process next in queue
                    j, q = queue[k].popleft()
                    s = t + (0 if lane == k else switch_over)
                    schedule[k].append(s) # record starting time
                    C = s + q # completion time
                    # add completion time event
                    last_completion_time = C
                    e = (C, -1, k, j)
                    heappush(h, e)
                    busy = True
                    lane = k
                    break

except IndexError:
    # heap has become empty
    pass


### compute the objective
obj = 0
res = {}
for k in range(K):
    # subtract, because cost
    obj -= sum(instance[f"length{k}"] * (np.array(schedule[k]) - instance[f"arrival{k}"]))

    res[f'start_time_{k}'] = np.array(schedule[k])
    res[f'end_time_{k}'] = np.array(schedule[k]) + instance[f'length{k}']

print(f"---- exhaustive objective = {obj}")

# save to file
np.savez(snakemake.output[0], obj=obj, K=K, **res)
