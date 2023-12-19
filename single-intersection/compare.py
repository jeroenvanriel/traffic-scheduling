import numpy as np
from collections import defaultdict
from glob import glob
import re


if __name__ == "__main__":

    objectives = defaultdict(list)
    times = defaultdict(list)
    ids = set()

    # get all exact solutions
    for in_file in glob("./schedules/exact_*.npz"):
        m = re.match(r".\/schedules\/exact\_(\d+)\_(\d+)\.npz", in_file)
        if m is None:
            print(in_file)
            raise Exception("Incorrect instance file name.")

        ix = m.group(1) # experiment number
        ids.add(ix)

        p = np.load(in_file)
        objectives[ix].append(p['obj'])
        times[ix].append(p['time'])


    for ix in sorted(ids):
        obj = sum(objectives[ix]) / len(objectives[ix])
        t = sum(times[ix]) / len(times[ix])
        n = len(times[ix])
        print(f"instance {ix:>2}: obj = {obj:.4f}, " \
            f"time = {t:.4f} " \
            f"({n} samples)")
