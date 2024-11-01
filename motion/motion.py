import numpy as np
import pandas as pd
from amplpy import AMPL
import matplotlib.pyplot as plt


def motion_synthesize(checkpoints, params, prev=None):
    """MotionSynthesize with N checkpoints, specified as a numpy array with
    shape (N,2). The second axis contains (time, position) pairs."""
    model = r"""
    param T; # number of discrete time epochs
    param dt; # discrete time step size
    set Ts = 0..T-1; # epoch indices

    param C; # number of checkpoints
    set Cs = 1..C;
    param c_t {Cs};
    param c_pos {Cs};

    param vmax; # maximum velocity
    param umax; # maximum control

    param l; # minimum follow distance

    # state and control variables
    var x {Ts}; # position
    var v {Ts}; # velocity
    var u {Ts}; # control (acceleration)

    maximize objective: sum {t in Ts} abs(x[t]);

    subject to x_checkpoint {c in Cs}: x[c_t[c]] = c_pos[c];
    subject to v_checkpoint {c in Cs}: v[c_t[c]] = vmax;

    subject to v_bounds {t in Ts}: 0 <= v[t] <= vmax;
    subject to u_bounds {t in Ts}: abs(u[t]) <= umax;

    # forward Euler
    subject to integrate_x {t in 0..T-2}: x[t+1] = x[t] + dt*v[t];
    subject to integrate_v {t in 0..T-2}: v[t+1] = v[t] + dt*u[t];
    """

    # add follow constraint for vehicle ahead
    if prev is not None:
        model += r"""
        param Y; # number of discrete time steps for trajectory ahead
        set Ys = 1..Y;
        param y {Ys}; # position of vehicle in front
        subject to follow {t in Ys}: y[t] - x[t] >= l;"""

    ampl = AMPL()
    ampl.eval(model)

    # first and final time step indices
    dt = params["dt"]
    t0_i = int(checkpoints[ 0, 0] / dt)
    tf_i = int(checkpoints[-1, 0] / dt)
    ampl.param["T"] = (tf_i - t0_i) + 1

    if prev is not None:
        # need to align the previous trajectory
        prev_tf = prev[0]
        Y = int((prev_tf - checkpoints[0, 0]) / dt)
        ampl.param["Y"] = Y
        ampl.param["y"] = prev[1][-Y:]

    # checkpoints
    ampl.param["C"] = checkpoints.shape[0]
    ampl.param["c_t"] = [int(t / dt) - t0_i for t in checkpoints[:,0]]
    ampl.param["c_pos"] = checkpoints[:,1]

    for key, value in params.items():
        ampl.param[key] = value

    ampl.solve(solver="scip")
    assert ampl.solve_result == "solved"
    return ampl.get_variable("x").to_pandas().T.values.tolist()[0]


def generate_trajectories(instance, y, params):
    """From a crossing time schedule y, generate trajectories using the
    MotionSynthesize procedure."""
    nodes = instance['G'].nodes
    route = instance['route']
    release = instance['release']

    N = len(release) # number of classes
    n = [len(r) for r in release] # number of arrivals per class

    trajectories = [[] for l in range(N)]

    for l in range(N):
        for k in range(n[l]):
            pos_cum = 0 # cumulative position
            checkpoints = np.empty((len(route[l]), 2))
            # first checkpoint is just (t = release time, relative position = 0)
            checkpoints[0] = np.array([release[l][k], 0])
            for i in range(1, len(route[l])):
                u = route[l][i-1]
                v = route[l][i]
                t = y[l, k, v]
                pos_cum += float(np.linalg.norm(np.array(nodes[u]['pos']) - np.array(nodes[v]['pos'])))
                checkpoints[i] = np.array([t, pos_cum])
            prev = None
            if k > 0: # there is a vehicle ahead
                prev = (prev_tf, trajectories[l][-1][1])
            traject = motion_synthesize(checkpoints, params, prev=prev)
            trajectories[l].append((checkpoints[0, 0], traject))
            prev_tf = checkpoints[-1, 0] # final time of previous trajectory

    return trajectories
