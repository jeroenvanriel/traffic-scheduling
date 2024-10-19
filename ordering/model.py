import numpy as np
import torch
import torch.nn as nn


class PaddedEmbeddingModel(nn.Module):
    """Should be used with 2 lanes, because of how actions are transformed to lanes."""

    def __init__(self, in_shape):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(in_shape).prod(), 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x)

    def state_transform(automaton, horizon=5):
        # compute minimum LB of unscheduled vehicles
        LBs = []
        for LB_lane, k_lane in zip(automaton.LB, automaton.k):
            LBs.extend(LB_lane[k_lane:])
        min_LB = min(LBs)

        # create the observation padded with zeros
        obs = np.zeros((len(automaton.LB), horizon))
        for l, (LB_lane, k_lane) in enumerate(zip(automaton.LB, automaton.k)):
            actual_horizon = min(horizon, len(LB_lane) - k_lane)
            obs[l,:actual_horizon] = LB_lane[k_lane:k_lane+actual_horizon] - min_LB

        # lane cycling
        last_lane = automaton.last_lane
        if last_lane is None:
            last_lane = 0 # assume initial last_lane == 0
        obs = np.roll(obs, last_lane, axis=0)

        return torch.as_tensor(obs.flatten(), dtype=torch.float, device=torch.device('cuda'))

    def inverse_action_transform(automaton, lane):
        # stay on last_lane when action == 0
        # switch to other lane when action == 1
        if automaton.last_lane == None:
            # assume initial last_lane == 0
            out = lane
        else:
            out = int(automaton.last_lane != lane)

        return torch.as_tensor(out, dtype=torch.float, device=torch.device('cuda'))

    def action_transform(automaton, logit):
        action = logit > 0
        action = int(action.detach().cpu().numpy()[0])

        if automaton.last_lane == None:
            lane = action
        else:
            lane = abs(int(automaton.last_lane - action))

        if automaton.k[lane] == automaton.K[lane]:
            # chosen lane is done, so choose other lane
            lane = abs(int(1 - lane))

        return lane
