import numpy as np
import pandas as pd
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
        continue

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

# Load experiment specification from .csv file. Check which parameter to plot on
# the x-axis, assuming only one column changes. Whenever more than one parameter
# changes, we would require either multiple plots or a visualization like
# parallel coordinate plot.
specs_file = snakemake.input['spec_file']
specs = pd.read_csv(specs_file)
# count unique values per column
changing_col = None
for col in specs:
    if len(specs[col].unique()) > 1:
        if changing_col is not None:
            raise Exception("Multiple columns with changing values in experiment"
                            " specification not supported.")
        changing_col = col

x_values = specs[changing_col]

# choose the appropriate plot function
plot = plt.plot if len(x_values) > 1 else plt.scatter

for policy in policies:
    plot(x_values, scores[policy], label=policy)

plt.legend()
plt.savefig(snakemake.output[0])
