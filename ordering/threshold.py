def threshold_heuristic(automaton, tau=0):
    # default first lane is 0
    l = automaton.last_lane or 0
    # last scheduled vehicle
    k = automaton.k[l] - 1

    # next vehicle exists
    if k + 1 < automaton.K[l]:
        # earliest crossing time of next vehicle in lane
        r = automaton.instance['release'][l][k+1]

        if automaton.LB[l][k] + automaton.instance['length'][l][k] + tau >= r:
            return l

    # next lane with unscheduled vehicles
    l = (l + 1) % automaton.N
    while automaton.k[l] == automaton.K[l]:
        l = (l + 1) % automaton.N
    return l
