import numpy as np
import os
from ppo import train

# Dynamically load the python module with the generators.
generator_code = snakemake.input[1]
module = os.path.splitext(os.path.basename(generator_code))[0]
instance_generator = __import__(module).instance_generator

# specification file
infile = snakemake.input[0]
spec = np.load(infile)

# horizon
horizon = snakemake.params['horizon']

# target
model_path = snakemake.output[0]

# Train DQN by sampling from problem distribution.
train(model_path, spec['K'], lambda: instance_generator(spec), spec['s'], horizon)
