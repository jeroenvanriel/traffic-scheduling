from util import vehicle_indices, route_indices, order_indices, routes_at_intersection, pos_along_route, dist
import networkx as nx
from networkx import topological_sort
from collections import defaultdict
from matplotlib import colormaps


def next_intersection(route, v):
    """Get next intersection after v on route."""
    ix = route.index(v)
    if ix + 1 < len(route):
        return route[ix + 1]


class DisjunctiveGraph(nx.DiGraph):

    def remove_done_edges(self):
        """Remove all edges from or to nodes that are done."""
        to_remove = [edge for x, y in self.nodes(data=True) if y['done']==1 for
                     edge in [*self.out_edges(x), *self.in_edges(x)]]
        self.remove_edges_from(to_remove)


    def draw(self, intersection=None):
        pos = {}
        nodes = []
        for r, k, v in self.nodes:
            if intersection is None or v == intersection:
                nodes.append((r, k, v))
                pos[r, k, v] = (k, r)

        nx.draw_networkx(self.subgraph(nodes), pos=pos, with_labels=False,
                         node_size=1600, arrowsize=20)

        # indices
        labels = { (r, k, v): f"{r}: {k}\n{v}" for r, k, v in self.subgraph(nodes) }
        nx.draw_networkx_labels(self.subgraph(nodes), labels=labels,
                                font_size=9, pos={ i: (pos[i][0], pos[i][1]) for i in pos })


class Automaton:
    """Dynamically updates disjunctive graph augmented with crossing time lower
    bounds. Assumes non-negative crossing times.

    Automaton.D is a networkx graph representing the disjunctive graph. Each
    node has the following attributes:

    - `LB` contains the crossing time lower bound.
    - `done` (0 or 1) indicates whether the operation has been scheduled.
    - `action_mask` indicates whether this node encodes an operation that can be
      scheduled next, i.e, a valid action, thus encoding the action space mask."""

    def __init__(self, instance):
        self.instance = instance
        self.G = instance['G']
        self.route = instance['route']

        self.indices = vehicle_indices(instance)
        self.route_indices = route_indices(self.indices) # "r"
        self.order_indices = order_indices(self.indices) # "k"
        self.crossing_indices = [ (r, v) for r in self.route_indices for v in self.route[r][1:-1] ]
        self.routes_at_intersection = routes_at_intersection(instance)

        self.crossing_done = { (r, v): False for (r, v) in self.crossing_indices }
        self.done = False

        # local order for every intersection
        self.order = { v: [] for v in self.G.intersections }
        # route index r of last scheduled vehicle at intersection v
        self.last_scheduled_route = { v: None for v in self.G.intersections }
        # order index k of last scheduled vehicle of route r at intersection v
        self.last_scheduled_order = { (r, v): None for r, v in self.crossing_indices }
        # order indices ("k") of unscheduled vehicles for every route-intersection pair
        # (we use list(...) to create a copy)
        self.unscheduled = { (r, v): list(self.order_indices[r]) for r, v in self.crossing_indices }

        ### compute disjunctive graph for empty ordering ###

        self.D = DisjunctiveGraph()
        self.rho = instance['rho']
        self.sigma = instance['sigma']

        # nodes
        for r, k in self.indices:
            for v in self.route[r]:
                # set default lower bound zero, assuming non-negative crossing times
                self.D.add_node((r, k, v), label=str((r, k, v)), LB=0, done=0, action_mask=0)

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
                        if (capacity := self.G[v][w]['capacity']) >= 0:
                            k2 = k + capacity
                            if (r, k2) in self.indices:
                                rho_vw = capacity * self.rho - dist(self.G, v, w) / instance['vmax']
                                self.D.add_edge((r, k, w), (r, k2, v), weight=rho_vw)

        # reference to the conjunctive chains (used by recurrent embedding model)
        self.conjunctive_chains = {}
        for r, v in self.crossing_indices:
            self.conjunctive_chains[r, v] = [self.D.nodes[r, k, v] for k in self.unscheduled[r, v]]

        ### initialize attributes for empty ordering ###

        # set release dates as lower bounds on entry nodes
        # set done for operations on entry nodes
        for r, k in self.indices:
            v0 = self.route[r][0]
            self.D.nodes[r, k, v0]['LB'] = instance['release'][r][k]
            self.D.nodes[r, k, v0]['done'] = 1

        # compute lower bounds in remaining nodes
        self.update_LB()

        # set initial action space mask
        for r in self.route_indices:
            for v in self.route[r][1:-1]:
                self.D.nodes[r, 0, v]['action_mask'] = 1


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
            raise Exception("All vehicles in this route have already been scheduled at this intersection.")

        # pop from the start
        next_vehicle = self.unscheduled[r, v][0]
        del self.unscheduled[r, v][0]
        # append to local order
        self.order[v].append((r, next_vehicle))
        self.last_scheduled_route[v] = r
        self.last_scheduled_order[r, v] = next_vehicle

        # update done attribute in disjunctive graph
        self.D.nodes[r, next_vehicle, v]['done'] = 1

        # update action mask
        self.D.nodes[r, next_vehicle, v]['action_mask'] = 0
        if len(self.unscheduled[r, v]) > 0:
            # valid action becomes next operation of route r at v
            self.D.nodes[r, next_vehicle + 1, v]['action_mask'] = 1

        # add disjunctive arcs from (r0, k0) to first unscheduled (r1, k1) with r0 != r1 and intersecting routes
        r0, k0 = r, next_vehicle
        for r1 in self.route_indices:
            if r0 != r1 and v in self.route[r1] and len(self.unscheduled[r1, v]) > 0:
                k1 = self.unscheduled[r1, v][0] # first unscheduled vehicle of route r1
                self.D.add_edge((r0, k0, v), (r1, k1, v), weight=self.sigma)

        # compute LB for new partial schedule by updating in topological order
        self.update_LB()

        # check crossing done
        self.crossing_done[r, v] = len(self.unscheduled[r, v]) == 0
        # done when all crossings done
        self.done = all(self.crossing_done[r, v] for (r, v) in self.crossing_indices)



    def get_obj(self):
        """Compute the total objective, which is the sum of crossing times at
        all nodes that are not entry nodes."""
        obj = 0
        for r, k in self.indices:
            for v in self.route[r][1:]: # all but entry nodes
                obj += self.D.nodes[r, k, v]['LB']
        return obj
