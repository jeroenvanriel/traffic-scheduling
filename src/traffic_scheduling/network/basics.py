# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: traffic
#     language: python
#     name: python3
# ---

# %% [markdown]
# ### Preamble

# %% [markdown]
# > ⚠️ **Warning:** This notebook file is paired to a .py file with the same name, such that we can cleanly import the functionality from other notebooks. This is done using the facilities of the jupytext package. The cells in this notebook that are only meant as "demonstration" are marked with the cell tag "active-ipynb", which causes the jupytext synchronization command to ignore these when syncing to the .py file. This is our current way of doing "literate programming" with jupyter notebooks.

# %%
import numpy as np
import pandas as pd
import networkx as nx
import os
import matplotlib.pyplot as plt

# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# # Getting started

# %% [markdown]
# ## Network instance data class

# %%
from dataclasses import dataclass

@dataclass
class NetworkInstance:
    G: nx.DiGraph
    routes: list[np.ndarray] | None = None

    arrivals: list[np.ndarray] | None = None

    rho:   float = 1.2   # following time
    sigma: float = 1.7   # time between conflicting vehicles
    vmax:  float = 1     # maximum speed

    # parameters for trajectory generation (not required for scheduling)
    umax:  float = 1     # maximum acceleration
    veh_l: float = 1.2   # vehicle length
    veh_w: float = 0.5   # vehicle width

    @property
    def switch(self): return self.sigma - self.rho

    @property
    def R(self): return len(self.arrivals)
    @property
    def n(self): return [len(self.arrivals[r]) for r in range(self.R)]
    @property
    def N(self): return sum(self.n)

    @property
    def vehicle_indices(self):
        """Get list of vehicle indices {(r, k)} from instance specification,
        assuming consecutive numbering from 0 (so without gaps)."""
        R = range(self.R); n = [range(self.n[r]) for r in R]
        # use 'sorted' to guarantee lexicographical order
        return sorted([(r, k) for r in R for k in n[r]])


# %% [markdown]
# Connect some drawing function to this class, for convenient access.

# %%
from traffic_scheduling.network.drawing import draw_graph, draw_road

NetworkInstance.draw_graph = lambda self: draw_graph(self.G)
NetworkInstance.draw_road = lambda self, **kwargs: draw_road(self.G, **kwargs)

# %% [markdown]
# ## Optimal schedules with integer programming

# %% [markdown]
# ### Schedule data class

# %% [markdown]
# Plus schedule visualization, as Gantt chart.

# %%
from typing import Optional, Dict, Any

@dataclass
class NetworkMILPSchedule:
    instance: NetworkInstance
    # schedule is mapping from [route, vehicle, node] -> scheduled crossing time
    y: Dict[tuple[int, int, Any], float] 
    obj: Optional[float] = None
    done: Optional[bool] = None
    gap: Optional[float] = None
    time: Optional[float] = None
    progress: Optional[pd.DataFrame] = None

    @property
    def makespan(self):
        return max(self.y.values()) + self.instance.rho


# %% [markdown]
# Keep the schedule plot function local to this class, for convenience.

# %%
from traffic_scheduling.network.drawing import plot_schedule
NetworkMILPSchedule.plot_schedule = lambda self: plot_schedule(self)

# %% [markdown]
# ### Mixed-integer linear programming

# %% [markdown]
# > Note: this implementation uses Pyomo with SCIP. The previous `recordprogress` option has been removed because tracking MIP progress over time with Pyomo+SCIP is more involved compared to the previous implementation with Gurobi, and would require parsing solver logs afterward. Hence, the benchmark `notebooks/network/milp-benchmark-progress.ipynb` does not work anymore with the current implementation.

# %%
from pyomo.environ import *
from pyomo.core import ConcreteModel, Var, ConstraintList, Objective, NonNegativeReals, Binary, minimize, value
from pyomo.opt import SolverFactory, TerminationCondition
from itertools import product, combinations
from traffic_scheduling.network.util import dist

def solve(instance, gap=0.0, timelimit=0, consolelog=False, logfile=None):
    """Solve a network scheduling problem as a MILP."""
    routes = instance.routes
    arrivals = instance.arrivals

    rho = instance.rho
    sigma = instance.sigma

    R = instance.R # number of routes
    n = instance.n # number of arrivals per route

    # big-M
    M = 1000

    g = ConcreteModel()
    g.yvars = VarList = []  # placeholder to keep style close; actual vars stored in `y` dict
    g.conjunctions = ConstraintList()
    g.disjunctions = ConstraintList()
    g.distances = ConstraintList()
    g.buffers = ConstraintList()

    y = {}
    objective_terms = []

    # release time parameters and crossing time variables
    for l in range(R):
        for k in range(n[l]):
            for r in range(len(routes[l]) - 1): # skip the last node in each route, which is the exit point
                v = routes[l][r]
                if r == 0: # entrypoint
                    y[l, k, v] = arrivals[l][k]
                else: # intersections
                    y[l, k, v] = Var(domain=NonNegativeReals)
                    setattr(g, f"y_{l}_{k}_{r}", y[l, k, v])
                    objective_terms.append(y[l, k, v])

    # conjunctions...
    for l in range(R):
        for v in routes[l][1:-1]: # ...on all except the first and last node on each route
            for k in range(n[l] - 1):
                g.conjunctions.add(y[l, k, v] + rho <= y[l, k + 1, v])

    # disjunctions at route intersections
    for l1, l2 in combinations(range(R), 2):
        # intersections of routes is set of "merge points"
        for v in set(routes[l1]) & set(routes[l2]):
            for k1, k2 in product(range(n[l1]), range(n[l2])):
                oc = Var(domain=Binary)
                setattr(g, f"o_{l1}_{k1}_{l2}_{k2}", oc)

                g.disjunctions.add(y[l1, k1, v] + sigma <= y[l2, k2, v] + oc * M)
                g.disjunctions.add(y[l2, k2, v] + sigma <= y[l1, k1, v] + (1 - oc) * M)

    # distances
    for l in range(R):
        for k in range(n[l]):
            # skip the distance constraint to the exit point, we compute the entry time
            # at the exit point after solving
            for r in range(len(routes[l]) - 2):
                v = routes[l][r]
                w = routes[l][r + 1]
                g.distances.add(y[l, k, w] >= y[l, k, v] + dist(instance.G, v, w) / instance.vmax)

    # buffers
    for l in range(R):
        for r in range(len(routes[l]) - 1):
            v = routes[l][r]
            w = routes[l][r + 1]
            capacity = instance.G.edges[v, w]['capacity']
            if capacity == -1:
                continue
            for k in range(n[l] - capacity):
                rho_hat = capacity * rho - dist(instance.G, v, w) / instance.vmax
                g.buffers.add(y[l, k, w] + rho_hat <= y[l, k + capacity, v])

    g.obj = Objective(expr=sum(objective_terms), sense=minimize)

    solver = SolverFactory("scip")
    if not solver.available(False):
        raise RuntimeError(
            "SCIP executable not found. Install SCIP and ensure `scip` is on PATH to use this solver."
        )
    if gap > 0:
        solver.options["limits/gap"] = gap
    if timelimit > 0:
        solver.options["limits/time"] = timelimit
    if not consolelog:
        solver.options["display/verblevel"] = 0

    solve_kwargs = {"tee": consolelog}
    if logfile is not None:
        os.makedirs(os.path.dirname(os.path.abspath(logfile)), exist_ok=True)
        solve_kwargs["logfile"] = logfile

    results = solver.solve(g, **solve_kwargs)

    # compute the arrival time at the exit point for each vehicle, which is computed from the entry time at the last
    # intersection and the travel time
    for l in range(R):
        for k in range(n[l]):
            v = routes[l][-2]
            w = routes[l][-1]
            y[l, k, w] = value(y[l, k, v]) + dist(instance.G, v, w) / instance.vmax

    res = {}
    res['y'] = { k : (value(v) if hasattr(v, 'is_expression_type') or hasattr(v, 'is_variable_type') else v) for k, v in y.items() }
    res['obj'] = value(g.obj)
    res['done'] = int(results.solver.termination_condition == TerminationCondition.optimal)

    upper = getattr(results.problem, "upper_bound", None)
    lower = getattr(results.problem, "lower_bound", None)
    if upper is not None and lower is not None and np.isfinite(upper) and np.isfinite(lower):
        res['gap'] = abs(upper - lower) / (abs(upper) + 1e-10)
    elif res['done']:
        res['gap'] = 0.0
    else:
        res['gap'] = np.nan

    res['time'] = getattr(results.solver, "time", 0.0) or 0.0

    return NetworkMILPSchedule(instance=instance, **res)


# %% [markdown]
# Keep this method at the instance class, for convenience.

# %%
def solve_myself(self, **kwargs): self.opt = solve(self, **kwargs); return self.opt
NetworkInstance.solve = solve_myself

# %% [markdown]
# ## Instance generation

# %% [markdown]
# ### Network generation

# %%
from collections.abc import Iterable

def generate_grid_network(m=None, n=None, xdist=[10, 10], ydist=[10]):
    """Generate a grid network.

    Either specify:
    
    - number of columns (m) and rows (n) of intersection,
    with fixed horizontal (xdist) and vertical (ydist) spacing;

    - two lists of horizontal (xdist) and vertical (ydist) distances between
    consecutive intersections, such that there are `m = len(xdist) - 1` columns
    and `n = len(ydist) - 1` rows of intersections.

    Returns (G, routes), where G is a networkx graph and each route is a list of
    nodes. At the edges of the network, we also connect each intersection to an
    inbound/outbound node. Therefore, the total number of nodes is n*m +
    2*(n+m). Pure intersections nodes are stored in `network.intersections`. We
    generate all straight routes from west to east and from south to north."""

    G = nx.DiGraph()
    G.intersections = []

    assert m is not None or isinstance(xdist, Iterable), "Either specify m or provide xdist list."
    assert n is not None or isinstance(ydist, Iterable), "Either specify n or provide ydist list."
    if not isinstance(xdist, Iterable): xdist = [xdist] * (m+1)
    if not isinstance(ydist, Iterable): ydist = [ydist] * (n+1)
    if m is None: m = len(xdist) - 1
    if n is None: n = len(ydist) - 1

    def node(x, y):
        if (i == 0 or i == m+1) and (j == 0 or j == n+1): return # skip the corners
        G.add_node((i,j), pos=(x, y))
        # collect pure intersections (not inbound/outbound)
        if not (i == 0 or i == m+1) and not (j == 0 or j == n+1):
            G.intersections.append((i,j))

    # 1. Generate nodes
    # The node in the i'th row and j'th column is identified as (i,j).
    x = 0
    for i in range(m + 2):
        y = 0
        for j in range(n + 2):
            node(x, y)
            if j <= n: y += ydist[j]
        if i <= m: x += xdist[i]

    # 2. Add edges
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 and j == 0:
                continue # skip first corner

            # only both direction in the "interior"
            # negative capacity = unlimited
            if j != 0: G.add_edge((i,j), (i+1,j), capacity=-1, dist=xdist[i])
            if i != 0: G.add_edge((i,j), (i,j+1), capacity=-1, dist=ydist[j])

    # 3. Generate straight routes
    routes = [[] for _ in range(n+m)]
    for i in range(m + 2):
        for j in range(n + 2):
            if i != 0 and i != m+1: routes[i-1].append((i,j))   # west-east
            if j != 0 and j != n+1: routes[m+j-1].append((i,j)) # south-north

    return G, routes


# %% [markdown]
# ### Instance generation

# %% [markdown]
# Reuse the route arrival generation function from the single intersection.

# %%
from collections.abc import Iterable
from traffic_scheduling.single.basics import uniform, clipped, arrivals_from_gaps


def generate_instance(F, n, rho=1.2, sigma=1.7, net_m=3, net_n=2, distance=10):
    G, routes = generate_grid_network(net_m, net_n, distance, distance)
    R = len(routes)

    arrivals = []
    for r in range(R):
        # make explicit singletons
        if not callable(F): f = F[r]
        else: f = F
        if isinstance(n, Iterable): n1 = n[r]
        else: n1 = n

        # create random interarrival times ("gaps") and cumulate them to obtain
        # earliest arrival times
        arrivals.append(arrivals_from_gaps(f(n1), rho))

    return NetworkInstance(G, routes, arrivals, sigma=sigma, rho=rho)


def generate_simple_instance(n=[10, 10, 10, 10], net_m=2, net_n=2):
    return generate_instance(clipped(uniform()), n, net_m=net_m, net_n=net_n)


# %% [markdown]
# ## Disjunctive graphs for instances/schedules

# %% [markdown]
# The disjunctive graph is a useful tool for representing (partial) schedules. We include a function to construct the disjunctive graph for a given instance (empty schedule) and for a given complete schedule. It can also be useful for visualization for small instances, for which we also provide a utility function.

# %%
import networkx as nx
from traffic_scheduling.network.util import dist

def next_intersection(route, v):
    """Get next intersection after v on route."""
    ix = route.index(v)
    if ix + 1 < len(route):
        return route[ix + 1]


def empty_disjunctive_graph(instance: NetworkInstance) -> nx.DiGraph:
    """Create the empty disjunctive graph for the given instance."""
    routes = instance.routes
    indices = instance.vehicle_indices
    route_indices = range(instance.R)
    order_indices = { route: range(instance.n[route]) for route in route_indices }

    D = nx.DiGraph()

    # nodes
    for r, k in indices:
        for v in routes[r]:
            # set default lower bound zero, assuming non-negative crossing times
            D.add_node((r, k, v), label=str((r, k, v)), LB=0, done=0, action_mask=0)

    # edges
    for r in route_indices:
        for v in routes[r]:
            for k in order_indices[r]:
                if k + 1 < len(order_indices[r]):
                    # conjunction
                    D.add_edge((r, k, v), (r, k + 1, v), weight=instance.rho)

                if (w := next_intersection(routes[r], v)) is not None:
                    # travel constraint
                    D.add_edge((r, k, v), (r, k, w), weight=dist(instance.G, v, w) / instance.vmax)

                    # buffer constraint
                    if (capacity := instance.G[v][w]['capacity']) >= 0:
                        k2 = k + capacity
                        if (r, k2) in indices:
                            rho_vw = capacity * instance.rho - dist(instance.G, v, w) / instance.vmax
                            D.add_edge((r, k, w), (r, k2, v), weight=rho_vw)

    return D


def full_disjunctive_graph(schedule: NetworkMILPSchedule) -> nx.DiGraph:
    """Create the full disjunctive graph for the given schedule."""
    D = empty_disjunctive_graph(schedule.instance)

    intersections = { v for route in schedule.instance.routes for v in route[1:-1] }

    indices_at_intersection = { v: [] for v in intersections }
    for r, k in schedule.instance.vehicle_indices:
        for v in schedule.instance.routes[r][1:-1]: # exclude entry/exit nodes
            indices_at_intersection[v].append((r, k))

    rho = schedule.instance.rho
    sigma = schedule.instance.sigma

    # add the directed disjunctive edges based on the schedule
    for v in intersections:
        # sort by scheduled crossing time
        indices = indices_at_intersection[v]
        indices.sort(key=lambda rk: schedule.y[rk[0], rk[1], v])
        for (r1, k1), (r2, k2) in zip(indices[:-1], indices[1:]):
            D.add_edge((r1, k1, v), (r2, k2, v), weight=rho if r1 == r2 else sigma)

    return D


# %%
def draw_disjunctive_graph(D, intersection=None):
    pos = {}
    nodes = []
    for r, k, v in D.nodes:
        if intersection is None or v == intersection:
            nodes.append((r, k, v))
            pos[r, k, v] = (k, r)

    nx.draw_networkx(D.subgraph(nodes), pos=pos, with_labels=False,
                        node_size=1600, arrowsize=20)

    # indices
    labels = { (r, k, v): f"{r}: {k}\n{v}" for r, k, v in D.subgraph(nodes) }
    nx.draw_networkx_labels(D.subgraph(nodes), labels=labels,
                            font_size=9, pos={ i: (pos[i][0], pos[i][1]) for i in pos })


# %% [markdown]
# ## Schedule regularization

# %% [markdown]
# Before using a schedule for motion synthesis, we need to align the schedule times to regular time steps. This is done by rounding up the schedule times to the nearest multiple of the time step and checking the constraints. By visiting the nodes of the disjunctive graph in topological order, we only need one round of rounding.

# %%
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class NetworkStepSchedule:
    original: NetworkMILPSchedule
    # mapping from (route, vehicle, node) to time step index
    steps: Dict[tuple[int, int, Any], int]
    y: Dict[tuple[int, int, Any], float]
    time_step: float

def regularize_schedule(schedule: NetworkMILPSchedule, time_step: float) -> NetworkStepSchedule:
    """Regularize the schedule times to be multiples of the time step."""
    steps = { (r, k, v): None for r, k, v in schedule.y.keys() }
    y = schedule.y.copy()

    # compute full disjunctive graph, compute topological order of nodes
    D = full_disjunctive_graph(schedule)
    topological_order = list(nx.topological_sort(D))

    # update in topological order of disjunctive graph to ensure single-pass
    for node in topological_order:
        preds = list(D.predecessors(node))
        if preds:
            max_pred_time = max(y[pred] + D.edges[pred, node]['weight'] for pred in preds)
            y[node] = max(y[node], max_pred_time)

        # round up to nearest multiple of time_step
        # NOTE: might need a small tolerance here
        steps[node] = int((y[node] + time_step - 1e-9) // time_step)

    return NetworkStepSchedule(original=schedule, steps=steps, y=y, time_step=time_step)


# %%
from traffic_scheduling.network.drawing import plot_schedule

def plot_step_schedule(step_schedule: NetworkStepSchedule):
    """Plot the step schedule by converting it back to a regular schedule."""
    dummy_schedule = NetworkMILPSchedule(
        instance=step_schedule.original.instance,
        y={ key: step * step_schedule.time_step for key, step in step_schedule.steps.items() },
    )
    plot_schedule(dummy_schedule)

NetworkStepSchedule.plot_schedule = lambda self: plot_step_schedule(self)

# %% [markdown]
# ## Testing

# %% [markdown]
# ### Solving

# %% tags=["active-ipynb"]
# F = clipped(uniform(1, 3), min=0.1)
# instance = generate_instance(F, n=[3,3], net_m=1, net_n=1)
# instance.solve(timelimit=10);
# # instance.draw_graph();
# # instance.draw_road();

# %% tags=["active-ipynb"]
# instance.opt.plot_schedule()

# %% [markdown]
# ### Trajectories from scheduled crossing times

# %% tags=["active-ipynb"]
# from traffic_scheduling.motion import generate_trajectories
# params = dict(vmax=1, umax=1, l=1.19, dt=0.25)
# trajectories = generate_trajectories(instance.G, instance.routes, instance.arrivals, instance.opt.y, 1.2, params)

# %% tags=["active-ipynb"]
# def plot_trajectories(trajectories, dt):
#     for r in range(len(trajectories)):
#         for k in range(len(trajectories[r])):
#             t0 = trajectories[r][k][0]
#             t = [t0 + i*dt for i in range(len(trajectories[r][k][1]))]
#             plt.plot(t, trajectories[r][k][1], 'k')
#         plt.show()

# %% tags=["active-ipynb"]
# plot_trajectories(trajectories, 0.25)

# %%
