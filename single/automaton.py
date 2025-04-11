import numpy as np

from util import LB


class Automaton:

    def __init__(self, instance):
        self.instance = instance
        self.done = False
        # number of lanes
        self.N = len(instance['release'])
        # number of available vehicles per lane
        self.K = np.array([len(instance['release'][l]) for l in range(self.N)])
        # number of scheduled vehicles per lane
        self.k = np.array([0 for _ in range(self.N)])
        # initial empty schedule
        self.y = [[] for _ in range(self.N)]
        # compute LBs for empty schedule
        self.LB = LB(instance)
        # keep track of last lane
        self.last_lane = None


    def step(self, lane):
        # check if lane is done
        if self.k[lane] == self.K[lane]:
            raise Exception("All vehicles in this lane have already been scheduled.")

        # set y_i = LB(i)
        y = self.LB[lane][self.k[lane]]
        self.y[lane].append(y)

        # update number of scheduled vehicles
        self.k[lane] += 1

        prev_obj = sum(sum(self.LB[lane]) for lane in range(self.N))

        # update LB according to new partial schedule
        self.LB = LB(self.instance, {'y': self.y})

        # done when all vehicles have been scheduled
        self.done = (self.k == self.K).all()

        self.last_lane = lane

        # reward is change in partial objective (negative)
        return prev_obj - sum(sum(self.LB[lane]) for lane in range(self.N))


    def exhaustive(self, lane):
        """Whether the exhaustive service rule applies to lane."""
        if self.k[lane] == self.K[lane]:
            return False # this lane is done

        i = self.k[lane] - 1  # last scheduled
        j = self.k[lane]      # next unscheduled

        if i < 0: # no vehicles scheduled yet
            return False

        return self.LB[lane][i] + self.instance['length'][lane][i] == self.LB[lane][j]


    def get_objective(self):
        return sum(sum(ys) for ys in self.y)


def evaluate(instance, model):
    automaton = Automaton(instance)
    while not automaton.done:
        lane = model(automaton)
        automaton.step(lane)
    return automaton.get_objective()
