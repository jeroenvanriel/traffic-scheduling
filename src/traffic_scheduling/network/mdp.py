from traffic_scheduling.network.util import vehicle_indices, route_indices, order_indices, routes_at_intersection
from traffic_scheduling.network.basics import empty_disjunctive_graph
import networkx as nx


class NetworkScheduleEnv:
    """Dynamically updates disjunctive graph augmented with crossing time lower
    bounds. Assumes non-negative crossing times.

    NetworkScheduleEnv.D is a networkx graph representing the disjunctive graph. Each
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

        self.pending_intersections = self.G.intersections.copy()
        self.pending_crossings = self.crossing_indices.copy()
        # unscheduled vehicle order indices at each crossing
        self.unscheduled = { (r, v): self.order_indices[r].copy() for r, v in self.crossing_indices }

        self.done = False

        self.last_intersection = None
        # last scheduled route r at intersection v
        self.last_route = { v: None for v in self.G.intersections }
        # last scheduled vehicle order k at crossing (r, v)
        self.last_order = { (r, v): None for r, v in self.crossing_indices }

        # local order for every intersection
        self.order = { v: [] for v in self.G.intersections }

        # number of vehicle-intersection pairs, or length of crossing sequence
        self.size = sum( len(route[1:-1]) * len(release)
                    for route, release in zip(self.route, self.instance['release']) )

        self.rho = instance['rho']
        self.sigma = instance['sigma']

        ### compute disjunctive graph for empty ordering ###

        self.D = empty_disjunctive_graph(self.instance)
        
         ### initialize attributes for empty ordering ###

        # set release dates as lower bounds on entry nodes
        # set done for operations on entry nodes
        for r, k in self.indices:
            v0 = self.route[r][0]
            self.D.nodes[r, k, v0]['LB'] = instance['release'][r][k]
            self.D.nodes[r, k, v0]['done'] = 1

        # compute lower bounds in remaining nodes
        self.update_LB()

        # store the sum of lower bounds at intersections (for computation of objective)
        self.beta0 = sum(self.D.nodes[r, k, v]['LB']
                          for r in self.route_indices
                          for v in self.route[r][1:-1]
                          for k in self.order_indices[r])

        # set initial action space mask
        for r in self.route_indices:
            for v in self.route[r][1:-1]:
                self.D.nodes[r, 0, v]['action_mask'] = 1


    def update_LB(self):
        for v in nx.topological_sort(self.D):
            for u in self.D.predecessors(v):
                self.D.nodes[v]['LB'] = max(self.D.nodes[v]['LB'], self.D.nodes[u]['LB'] + self.D.edges[u, v]['weight'])


    def step(self, r, v):
        if v not in self.route[r][1:-1]:
            raise Exception(f"Node {v} is not an intersection on route {r}.")

        if (r, v) not in self.pending_crossings:
            raise Exception(f"Crossing {(r, v)} is already done.")

        prev_obj = self.get_objective()

        # order index for current action
        k = self.unscheduled[r, v].pop(0)
        self.order[v].append((r, k))

        # record current action
        self.last_intersection = v
        self.last_route[v] = r
        self.last_order[r, v] = k

        # update pending crossings and intersections
        if len(self.unscheduled[r, v]) == 0:
            self.pending_crossings.remove((r, v))
        if all(((r, v) not in self.pending_crossings) for r in self.routes_at_intersection[v]):
            self.pending_intersections.remove(v)
        self.done = len(self.pending_crossings) == 0

        # update done attribute in disjunctive graph
        self.D.nodes[r, k, v]['done'] = 1

        # update action mask
        self.D.nodes[r, k, v]['action_mask'] = 0
        if (r, v) in self.pending_crossings:
            # valid action becomes next operation of route r at v
            self.D.nodes[r, k + 1, v]['action_mask'] = 1

        # add disjunctive arcs from (r0, k0) to first unscheduled (r1, k1) with r0 != r1 and intersecting routes
        r0, k0 = r, k
        for r1 in self.route_indices:
            if r0 != r1 and v in self.route[r1] and (r1, v) in self.pending_crossings:
                k1 = self.unscheduled[r1, v][0] # first unscheduled vehicle of route r1
                self.D.add_edge((r0, k0, v), (r1, k1, v), weight=self.sigma)

        # compute LB for new partial schedule by updating in topological order
        self.update_LB()

        # reward is change in partial objective (negative)
        return prev_obj - self.get_objective()


    def get_objective(self):
        """Compute the total objective, which is the sum of delays at all
        intersections, divided by the total number of vehicle-intersection
        pairs."""
        obj = 0
        for r, k in self.indices:
            for v in self.route[r][1:-1]: # at all intersections
                obj += self.D.nodes[r, k, v]['LB']

        # subtract the initial sum of lower bounds beta0
        return (obj - self.beta0) / self.size
