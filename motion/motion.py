import numpy as np
import pandas as pd
from amplpy import AMPL
import matplotlib.pyplot as plt


def motion_synthesize(C_t, C_pos, params, prev=None):
    """MotionSynthesize with multiple checkpoints, specified as a
    list of times C_t and a list of positions C_pos."""
    model = r"""
    param T; # number of discrete time epochs
    param dt; # discrete time step size
    set Ts = 0..T-1; # epoch indices

    param C; # number of checkpoints
    set Cs = 1..C;
    param C_t {Cs};
    param C_pos {Cs};

    param vmax; # maximum velocity
    param umax; # maximum control

    param l; # minimum follow distance

    # state and control variables
    var x {Ts}; # position
    var v {Ts}; # velocity
    var u {Ts}; # control (acceleration)

    maximize objective: sum {t in Ts} abs(x[t]);

    subject to x_checkpoint {c in Cs}: x[C_t[c]] = C_pos[c];
    subject to v_checkpoint {c in Cs}: v[C_t[c]] = vmax;

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
    t0_i = int(C_t[0] / dt)
    tf_i = int(C_t[-1] / dt)
    ampl.param["T"] = (tf_i - t0_i) + 1

    if prev is not None:
        # need to align the previous trajectory
        prev_tf = prev[0]
        Y = int((prev_tf - C_t[0]) / dt)
        ampl.param["Y"] = Y
        ampl.param["y"] = prev[1][-Y:]

    # checkpoints
    ampl.param["C"] = len(C_t)
    ampl.param["C_t"] = [int(t / dt) - t0_i for t in C_t]
    ampl.param["C_pos"] = C_pos

    for key, value in params.items():
        ampl.param[key] = value

    ampl.solve(solver="scip")
    assert ampl.solve_result == "solved"
    return ampl.get_variable("x").to_pandas()
