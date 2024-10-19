import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import colormaps
from copy import deepcopy
from itertools import product


def plot_schedule(instance, schedule=None, out=None,
                  start_at_1=False, draw_instance=True, draw_switch=False, custom_end_time=None, clean=False):
    """Plot (partial) schedule given some instance.

    Lanes are presented from top to bottom. Use the `start_at_1` flag for 1-based
    lane numbering. When a (partial) schedule is provided, it is shown in the bottom row."""
    N = (len(instance['release']) if draw_instance else 0) \
        + (1 if schedule is not None else 0) # number of rows in the figure
    end = end_time(instance, schedule) if custom_end_time is None else custom_end_time
    height = 0.7 # row height
    y_scale = 0.7 # horizontal scaling
    fig, ax = plt.subplots(figsize=(end, 1 + y_scale * (N-1)))
    cmap = colormaps["tab10"] # lane colors
    if clean:
        cmap = colormaps["Set3"]

    # instance per lane
    if draw_instance:
        for l, (release, length) in enumerate(zip(instance['release'], instance['length'])):
            for r, p in np.nditer([release, length]):
                ax.add_patch(Rectangle((r, N-l-1 - height / 2), width=p, height=height,
                                    linewidth=1, facecolor=cmap(l), edgecolor='k'))

    # schedule
    if schedule is not None:
        for l, ys in enumerate(schedule['y']):
            if len(ys) == 0:
                continue # no scheduled vehicles in this lane
            for y, p in np.nditer([ys, instance['length'][l][:len(ys)]]):
                ax.add_patch(Rectangle((y, 0 - height / 2), width=p, height=height,
                                       linewidth=1, facecolor=cmap(l), edgecolor='k'))

    # switch-over arrows
    if draw_switch:
        pi = vehicle_order(schedule)
        prev_l, prev_k = pi[0][0], pi[0][1]
        for (l, k) in pi[1:]:
            if l != prev_l:
                origin = schedule['y'][prev_l][prev_k] + instance['length'][prev_l][prev_k]
                plt.arrow(origin, 0, instance['switch'], 0,
                          head_length=0.1, head_width=0.1, length_includes_head=True, color='k')
            prev_l, prev_k = l, k

    ax.set_xlim([-0.05, end + 0.05])
    ax.margins(y=0.15 / N)

    ticks = np.arange(N)
    lane_labels = np.arange(int(start_at_1), N + int(start_at_1))
    if schedule is not None:
        labels = ['y', *np.flip(lane_labels[:-1])] # top to bottom numbering
    else:
        labels = np.flip(lane_labels)
    plt.yticks(ticks=ticks, labels=labels)
    if clean:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(out) if out is not None else plt.show()
    plt.close()



def assert_valid_instance(instance):
    """Check whether release and length specify non-overlapping vehicles for
    each lane. Overlap may exist between lanes, but not on the same lane."""
    for release, length in zip(instance['release'], instance['length']):
        end_times = release + length
        end_times = np.roll(end_times, 1)
        end_times[0] = 0

        if not (release >= end_times).all():
            raise Exception("There are overlapping vehicles.")


def equalp(schedule1, schedule2):
    for y1, y2 in zip(schedule1['y'], schedule2['y']):
        if not np.allclose(y1, y2):
            return False
    return True


def LB_empty(instance):
    """Compute LB(i) in an empty schedule."""
    LBs = []
    for release, length in zip(instance['release'], instance['length']):
        # compute for this lane
        LB = np.empty_like(release)
        LB[0] = release[0]
        for i in range(1, len(release)):
            LB[i] = max(release[i], LB[i - 1] + length[i - 1])
        LBs.append(LB)

    return LBs


def LB(instance, schedule=None):
    """Compute LB(i) in a partial schedule."""
    if schedule is None:
        return LB_empty(instance)

    r = instance['release']

    # compute partial ordering
    pi = vehicle_order(schedule)

    # sequentially compute LB for scheduled vehicles
    LB = {}
    l, k = pi[0][0], pi[0][1]
    LB[l, k] = r[l][k]
    prev_l, prev_k = l, k
    for (l, k) in pi[1:]:
        w = instance['length'][prev_l][prev_k]
        if prev_l != l:
            w = w + instance['switch']
        LB[l, k] = max(r[l][k], LB[prev_l, prev_k] + w)
        prev_l, prev_k = l, k

    # compute LB of unscheduled vehicles by adapting r_i
    # and computing LB as if we had an empty schedule
    dummy = deepcopy(instance)

    # set r_i = LB(i) for scheduled vehicles
    for (l, k) in pi:
        dummy['release'][l][k] = LB[l, k]

    # consider arcs of each last scheduled vehicle of every lane
    for l1, l2 in product(range(len(r)), range(len(r))):
        k1 = len(schedule['y'][l1]) - 1 # last scheduled vehicle
        k2 = len(schedule['y'][l2]) # first unscheduled vehicle in other lane
        if k1 < 0 or k2 >= len(r[l2]): # they do not both exist?
            continue
        w = instance['length'][l1][k1] # edge weight
        if l1 != l2: # disjunctive arc
            w = w + instance['switch']
        dummy['release'][l2][k2] = max(dummy['release'][l2][k2], LB[l1, k1] + w)

    return LB_empty(dummy)


def vehicle_order(schedule):
    """Compute the vehicle order of a (partial) schedule given the crossing times."""
    indices = []
    values = {}
    N = len(schedule['y']) # number of lanes
    for l in range(N):
        K = len(schedule['y'][l]) # number of scheduled vehicles
        for k in range(K):
            indices.append((l, k))
            values[l, k] = schedule['y'][l][k]

    indices.sort(key=lambda i: values[i])
    return indices


def lane_order(schedule):
    """Compute the lane order of a (partial) schedule given the crossing times."""
    lanes = []
    for l, k in vehicle_order(schedule):
        lanes.append(l)
    return lanes


def end_time(instance, schedule=None):
    """Compute max(LB(i) + rho_i) in partial schedule."""
    lb = LB(instance, schedule)
    # we only have to consider each last vehicle per lane
    m = -np.inf
    for l in range(len(instance['release'])):
        t = lb[l][-1] + instance['length'][l][-1]
        if t > m:
            m = t
    return m
