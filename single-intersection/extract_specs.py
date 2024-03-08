import numpy as np
import pandas as pd

specs = pd.read_csv(snakemake.input[0])

for i, spec in specs.iterrows():
    # save to .npz file
    filename = snakemake.output[i]
    np.savez(filename, **spec)
