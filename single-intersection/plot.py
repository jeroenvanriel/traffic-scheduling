import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_data(data, xaxis='Epoch', value="AverageEpRet", group="Group", smooth=1, **kwargs):
    if smooth > 1:
        """
        smooth data with moving window average.
        that is,
            smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
        where the "smooth" param is width of that window (2k+1)
        """
        y = np.ones(smooth)
        for datum in data:
            x = np.asarray(datum[value])
            z = np.ones(len(x))
            smoothed_x = np.convolve(x,y,'same') / np.convolve(z,y,'same')
            datum[value] = smoothed_x

            print('last smoothed: ', smoothed_x[-1])

    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)
    sns.set(style="darkgrid", font_scale=1.5)
    sns.lineplot(data=data, x=xaxis, y=value, hue=group, errorbar="sd", **kwargs)

    plt.legend(loc='best').set_draggable(True)
    #plt.legend(loc='upper center', ncol=3, handlelength=1,
    #           borderaxespad=0., prop={'size': 13})

    """
    For the version of the legend used in the Spinning Up benchmarking page,
    swap L38 with:

    plt.legend(loc='upper center', ncol=6, handlelength=1,
               mode="expand", borderaxespad=0., prop={'size': 13})
    """

    xscale = np.max(np.asarray(data[xaxis])) > 5e3
    if xscale:
        # Just some formatting niceness: x-axis scale in scientific notation if max x is large
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    plt.tight_layout(pad=0.5)


if __name__ == "__main__":

    file_names = [
        "runs_data/SingleIntersectionEnv__dqn__1__1702994336.csv",
        "runs_data/SingleIntersectionEnv__dqn__2__1702994635.csv",
        "runs_data/SingleIntersectionEnv__dqn__3__1702994901.csv",
        "runs_data/SingleIntersectionEnv__dqn__4__1702995160.csv",
    ]

    datas = []

    for i, file_name in enumerate(file_names):
        # raw = np.loadtxt(file_name, delimiter=",", dtype=float, skiprows=1)
        data = pd.read_csv(file_name)

        # cols: wall time, step, value
        data = data.drop(columns=["Wall time"])
        data["Group"] = np.ones(len(data), dtype=int) * (i + 1)

        datas.append(data)

    plt.figure(figsize=(8, 6))
    plot_data(datas, xaxis="Step", value="Value", smooth=100)
    plt.savefig('comparison.pdf')
    plt.show()


    print(data)
