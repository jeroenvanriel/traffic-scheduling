import pandas as pd
import numpy as np
import math
from amplpy import AMPL
from network.util import vehicle_indices, order_indices


def motion_synthesize(T, p0=-10, v0=1, dt=0.1, vmax=1, amax=0.5, l=1, prev=None, **kwargs):
    """Solve the MotionSynthesize problem with direct transcription to a linear program.
    A part of the predecessor's trajectory can be specified by `prev`, which is a list
    of positions, starting from the first time epoch of this problem.

    T = time between arrival and departure from edge
    p0 = initial position
    vmax, amax = maximum speed and acceleration
    l = follow distance
    prev = previous trajectory, as list of positions at discrete time epochs from start
    """
    model = r"""
    param T; # number of discrete time epochs
    param dt; # discrete time epoch size
    set Ts = 1..T; # epoch indices

    param vmax; # maximum velocity
    param umax; # maximum control
    param p0; # initial position, must be < 0
    param v0; # initial velocity, must be 0 <= v0 <= 1
    param l; # minimum follow distance

    # state and control variables
    var x {Ts}; # position
    var v {Ts}; # velocity
    var u {Ts}; # control (acceleration)

    maximize objective: sum {t in Ts} x[t];

    subject to x_start: x[1] = p0;
    subject to v_start: v[1] = v0;
    #subject to x_end: x[T] = 0; # this is too restrictive numerically
    subject to x_end {t in Ts}: x[t] <= 0; # alternative
    subject to v_end: v[T] >= vmax - 0.01; # with a little tolerance

    subject to v_bounds {t in Ts}: 0 <= v[t] <= vmax;
    subject to u_bounds {t in Ts}: -umax <= u[t] <= umax;

    # forward Euler
    subject to integrate_x {t in 1..T-1}: x[t+1] = x[t] + dt*v[t];
    subject to integrate_v {t in 1..T-1}: v[t+1] = v[t] + dt*u[t];
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

    ampl.param["dt"] = dt
    ampl.param["T"] = T
    ampl.param["p0"] = p0
    ampl.param["v0"] = v0
    ampl.param["vmax"] = vmax
    ampl.param["umax"] = amax
    ampl.param["l"] = l

    if prev is not None:
        ampl.param["Y"] = len(prev)
        ampl.param["y"] = prev

    ampl.option['presolve_eps'] = 1e-12
    ampl.solve(solver="gurobi");
    assert ampl.solve_result == "solved"
    return ampl.get_variable("x").to_pandas().T.values.tolist()[0]


def generate_edge_trajectories(instance, y, dt, r, v):
    """From a crossing time schedule y, generate trajectories for edge (v, w) on
    route r using direct transcription (MotionSynthesize procedure). Node w is
    just the next node on the route. It is assumed that y also contains crossing
    times for entry and exit nodes.
    """
    # get the next node w
    route = instance['route'][r]
    w = route[route.index(v) + 1]
    # get distance of edge (v, w)
    D = instance['G'].edges[v, w]['dist']
    W = instance['W']

    # parameters for MotionSynthesize
    params = { 'dt': dt, 'l': instance['length'], 'v0': instance['vmax'],
               'vmax': instance['vmax'], 'amax': instance['amax'] }

    # now generate trajectories for edge (v, w)
    trajectories = {}
    prev = None
    for k in order_indices(vehicle_indices(instance))[r]:
        # discretize start and end times
        yv = int(math.ceil(y[r, k, v] / dt))
        yw = int(math.ceil(y[r, k, w] / dt))

        if prev is not None:
            # extract the relevant steps from prev trajectory
            prev = prev[prev['t'] >= yv * dt]['x'].to_numpy()

        # generate trajectory using direct transcription
        x = motion_synthesize(yw - yv + 1, p0=-D+W, prev=prev, **params)
        xd = pd.DataFrame(x, columns=['x'])
        # add corresponding time epochs
        xd['t'] = dt * np.array(range(yv, yw + 1))
        prev = xd.copy()
        trajectories[r, k] = xd

    return trajectories


def generate_route_trajectories(instance, y, dt, r):
    """Generate all trajectories for route r. Returns a map from vehicle index
    to trajectory as a pandas DataFrame with columns 't' and 'x'."""
    # glue together trajectories from generate_edge_trajectories()
    # with full speed parts in between
    pass


def generate_trajectories(instance, y, dt):
    """Generate trajectories for all routes. Returns a map from vehicle index to
    trajectory as a pandas DataFrame with columns 't' and 'x'."""
    pass
