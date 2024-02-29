import numpy as np
import os
from dqn import train

# load instances
infiles = snakemake.input[0:-1]
instances = []
for infile in infiles:
        instances.append(np.load(infile))

def gen():
    gen.i = (gen.i + 1) % len(instances)
    return instances[gen.i]

# static variable
gen.i = 0

# assumption: K and s are equal among instances
K = instances[0]['K']
s = instances[0]['s']

# horizon
horizon = snakemake.params['horizon']

# target
model_path = snakemake.output[0]

# Train DQN by sampling from problem distribution.
train(model_path, K, gen, s, horizon)
