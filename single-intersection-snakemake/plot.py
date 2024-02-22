import numpy as np
import matplotlib.pyplot as plt
import re
from collections import defaultdict

def average_score(files):
    scores = []
    for file in files:
        obj = np.load(file)['obj']
        scores.append(obj)

    return sum(scores) / len(scores)

# extract policy names from input files
policies = set()
for key in snakemake.input.keys():
    m = re.match(r"(\w+)\_(\d+)", key)
    if m is None:
        raise Exception("Incorrect file names (check Snakefile).")

    policy = m.group(1) # policy name
    policies.add(policy)

# for each combination of policy and experiment specification, average the
# performance scores over samples
n_specs = snakemake.params['n_specs']
scores = defaultdict(list)
for policy in policies:
    for i in range(n_specs):
        files = snakemake.input[f"{policy}_{i}"]
        scores[policy].append(average_score(files))

# plot the results per experiment
# TODO: load experiment specification from csv
rates = [2, 3, 4, 5] # corresponds to generate.arrival_params

for policy in policies:
    plt.plot(rates, scores[policy], label=policy)

plt.legend()
plt.savefig(snakemake.output[0])
