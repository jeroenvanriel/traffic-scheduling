import numpy as np
import matplotlib.pyplot as plt

n_specs = snakemake.params['n_specs']


def average_score(files):
    scores = []
    for file in files:
        obj = np.load(file)['obj']
        scores.append(obj)

    return sum(scores) / len(scores)


exactscores = []
dqnscores = []
exhaustivescores = []
simplescores = []
for i in range(n_specs):
    files = snakemake.input[f"exact_{i}"]
    exactscores.append(average_score(files))

    files = snakemake.input[f"dqn_{i}"]
    dqnscores.append(average_score(files))

    files = snakemake.input[f"exhaustive_{i}"]
    exhaustivescores.append(average_score(files))

    files = snakemake.input[f"simple_{i}"]
    simplescores.append(average_score(files))

# TODO: load from csv
rates = [2, 3, 4, 5] # corresponds to generate.arrival_params

plt.plot(rates, exactscores, label="exact")
#plt.plot(rates, simplescores, label="simple")
plt.plot(rates, exhaustivescores, label="exhaustive")
# TODO: different horizons
plt.plot(rates, dqnscores, label="$h^{(10)}$")

plt.legend()
plt.savefig(snakemake.output[0])
