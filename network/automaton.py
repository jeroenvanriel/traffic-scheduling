from util import vehicle_indices, route_indices, order_indices, pos_along_route, dist
import networkx as nx
from networkx import topological_sort
from collections import defaultdict


def next_intersection(route, v):
    """Get next intersection after v on route."""
    ix = route.index(v)
    if ix + 1 < len(route):
        return route[ix + 1]


class Automaton:
    """Dynamically updates disjunctive graph augmented with crossing time lower
    bounds.

    Automaton.D is a networkx graph representing the disjunctive graph. Each
    node has an `LB` attribute containing the crossing time lower bound and an
    `done` (0 or 1), indicating whether the operation has been scheduled.
    Assumes non-negative crossing times."""

    def __init__(self, instance):
        self.instance = instance
        self.G = instance['G']
        self.route = instance['route']
        self.done = False

        self.indices = vehicle_indices(instance)
        self.route_indices = route_indices(self.indices) # "r"
        self.order_indices = order_indices(self.indices) # "k"

        # local order for every intersection
        self.order = { v: [] for v in self.G.intersections }
        # order index k of last scheduled vehicle of route r at intersection v
        self.last_scheduled = { (r, v): None for r in self.route_indices for v in self.route[r][1:-1] }
        # order indices ("k") of unscheduled vehicles for every class-intersection pair
        # (we use list(...) to create a copy)
        self.unscheduled = { (r, v): list(self.order_indices[r]) for r in self.route_indices for v in self.route[r][1:-1] }

        ### compute disjunctive graph for empty ordering ###

        self.D = nx.DiGraph()
        self.rho = instance['rho']
        self.sigma = instance['sigma']

        # nodes
        for r, k in self.indices:
            for v in self.route[r]:
                # set default lower bound zero, assuming non-negative crossing times
                self.D.add_node((r, k, v), label=str((r, k, v)), LB=0, done=0)

        # edges
        for r in self.route_indices:
            for v in self.route[r]:
                for k in self.order_indices[r]:
                    if k + 1 < len(self.order_indices[r]):
                        # conjunction
                        self.D.add_edge((r, k, v), (r, k + 1, v), weight=self.rho)

                    if (w := next_intersection(self.route[r], v)) is not None:
                        # travel constraint
                        self.D.add_edge((r, k, v), (r, k, w), weight=dist(self.G, v, w) / instance['vmax'])

                        # buffer constraint
                        if capacity := self.G[v][w]['capacity'] >= 0:
                            k2 = k + capacity
                            if (r, k2) in self.indices:
                                rho_vw = capacity * self.rho - dist(self.G, v, w) / instance['vmax']
                                self.D.add_edge((r, k, w), (r, k2, v), weight=rho_vw)

        ### initialize attributes for empty ordering ###

        # set release dates and set done for initial operations
        for r, k in self.indices:
            v0 = self.route[r][0]
            self.D.nodes[r, k, v0]['LB'] = instance['release'][r][k]
            self.D.nodes[r, k, v0]['done'] = 1
        self.update_LB()

    def update_LB(self):
        for v in topological_sort(self.D):
            for u in self.D.predecessors(v):
                self.D.nodes[v]['LB'] = max(self.D.nodes[v]['LB'], self.D.nodes[u]['LB'] + self.D.edges[u, v]['weight'])


    def step(self, r, v):
        # check if v is an intersection on route of route r
        if v not in self.route[r] or self.route[r].index(v) in [0, len(self.route[r]) - 1]:
            raise Exception(f"Node {v} is not an intersection on the route of route {r}")

        # check if r has still unscheduled vehicles at v
        if len(self.unscheduled[r, v]) == 0:
            raise Exception("All vehicles in this class have already been scheduled at this intersection.")

        # pop from the start
        next_vehicle = self.unscheduled[r, v][0]
        del self.unscheduled[r, v][0]
        # append to local order
        self.order[v].append((r, next_vehicle))
        self.last_scheduled[r, v] = next_vehicle

        # update done attribute in disjunctive graph
        self.D.nodes[r, next_vehicle, v]['done'] = 1

        # add disjunctive arcs from (r0, k0) to first unscheduled (r1, k1) with r0 != r1 and intersecting routes
        r0, k0 = r, next_vehicle
        for r1 in self.route_indices:
            if r0 != r1 and v in self.route[r1] and len(self.unscheduled[r1, v]) > 0:
                k1 = self.unscheduled[r1, v][0] # first unscheduled vehicle of route r1
                self.D.add_edge((r0, k0, v), (r1, k1, v), weight=self.sigma)

        # compute LB for new partial schedule by updating in topological order
        self.update_LB()

        # done when all vehicles have been scheduled
        self.done = all(len(self.unscheduled[r, v]) == 0 for r in self.route_indices for v in self.route[r][1:-1])
