import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from glob import glob
import re


# Analyze the evaluation results and plot.
# (currently still rather ad-hoc)


## collect evaluation results from disk

ids = set()

exactscores = defaultdict(list)
for in_file in glob("./data/evaluation/*_exact_*.npz"):
    m = re.match(r".\/data/evaluation\/(\d+)\_exact\_(\d+)\.npz", in_file)
    if m is None:
        print(in_file)
        raise Exception("Incorrect instance file name.")

    ix = int(m.group(1)) # experiment number
    ids.add(ix)

    p = np.load(in_file)
    # N.B. the minus sign
    exactscores[ix].append( - float(p['obj']))

exactscores = dict(sorted(exactscores.items())) # sort by key
exactscores = list(map(np.mean, map(np.array, exactscores.values())))

dqnscores = defaultdict(list)
for ix in sorted(ids):
    for horizon in [10, 20, 30]:
        in_file = f"./data/evaluation/{ix}_dqn_horizon{horizon}.npz"

        p = np.load(in_file)
        dqnscores[horizon].append(p['scores'].mean())


## plot results

rates = [2, 3, 4, 5] # corresponds to generate.arrival_params

plt.plot(rates, exactscores, label="exact")
plt.plot(rates, dqnscores[10], label="$h^{(10)}$")
plt.plot(rates, dqnscores[20], label="$h^{(20)}$")
plt.plot(rates, dqnscores[30], label="$h^{(30)}$")
plt.legend()
plt.xlabel("arrival rate")
plt.xticks(rates)

plt.savefig("compare.pdf")
