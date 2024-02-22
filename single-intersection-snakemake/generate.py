import numpy as np
import pandas as pd
import os

module = os.path.splitext(os.path.basename(snakemake.input[1]))[0]

# Dynamically load the python module with the generator.
instance_generator = __import__(module).instance_generator

spec = np.load(snakemake.input[0])

for outfile in snakemake.output:
    res = instance_generator(spec)
    np.savez(outfile, K=spec['K'], s=spec['s'], **res)
