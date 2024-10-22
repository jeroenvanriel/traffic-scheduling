import numpy as np
import pandas as pd
from amplpy import AMPL
import matplotlib.pyplot as plt


model = r"""
param T; # number of discrete time steps
param dt; # discrete time step size
set Ts = 0..T; # step indices

param x0; # initial position
param v0; # initial velocity
param vmax; # maximum/final velocity
param umax; # maximum control

param y {Ts}; # position of vehicle in front
param l; # minimum follow distance

# state and control variables
var x {Ts}; # position
var v {Ts}; # velocity
var u {Ts}; # control (acceleration)

minimize objective: sum {t in Ts} abs(x[t]);

subject to x_initial : x[0] = x0;
subject to v_initial : v[0] = v0;

subject to x_final: x[T] = 0;
subject to v_final: v[T] = vmax;

subject to v_bounds {t in Ts}: 0 <= v[t] <= vmax;
subject to u_bounds {t in Ts}: abs(u[t]) <= umax;

# forward Euler
subject to integrate_x {t in 0..T-1}: x[t+1] = x[t] + dt*v[t];
subject to integrate_v {t in 0..T-1}: v[t+1] = v[t] + dt*u[t];

subject to follow {t in Ts}: abs(x[t] - y[t]) >= l;
"""


def motion_synthesize(tf, dt, x0, v0, vmax, umax, l, y):
    ampl = AMPL()
    ampl.eval(model)

    ampl.param["T"] = int(tf / dt)
    ampl.param["dt"] = dt
    ampl.param["x0"] = x0
    ampl.param["v0"] = v0
    ampl.param["vmax"] = vmax
    ampl.param["umax"] = umax
    ampl.param["l"] = l
    ampl.param["y"] = y

    ampl.solve(solver="scip")
    assert ampl.solve_result == "solved"
    return ampl.get_variable("x").to_pandas()


# some example trajectory y
t = np.arange(0, tf + dt, dt)
y = -15 + t

tf = 50
dt = 0.5

x0 = -20
v0 = 0
vmax = 2
umax = 1
l = 1

x = motion_synthesize(tf, dt, x0, v0, vmax, umax, l, y)

plt.plot(t, y)
plt.plot(t, x)
plt.show()
