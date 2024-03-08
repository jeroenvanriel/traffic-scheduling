import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


infile = snakemake.input[0]
schedule = np.load(infile)
K = int(schedule['K'])

fig, ax = plt.subplots(figsize=(20, 1))

height = 0.7
for k in range(K):
    starts = schedule[f'start_time_{k}']
    ends = schedule[f'end_time_{k}']

    for start, end in np.nditer([starts, ends]):
        ax.add_patch(Rectangle((start, k - height / 2), width=end-start, height=height))

ax.autoscale()
plt.yticks(np.arange(K))
plt.tight_layout()
plt.savefig(snakemake.output[0])
