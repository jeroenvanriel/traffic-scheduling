import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

infile = snakemake.input[0]
instance = np.load(infile)
K = int(instance['K'])

fig, ax = plt.subplots(figsize=(20, 1))

height = 0.7
for k in range(K):
    arrivals = instance[f"arrival{k}"]
    lengths = instance[f"length{k}"]

    for arrival, length in np.nditer([arrivals, lengths]):
        ax.add_patch(Rectangle((arrival, k - height / 2), width=length, height=height))

ax.autoscale()
plt.yticks(np.arange(K))
plt.tight_layout()
plt.savefig(snakemake.output[0])
