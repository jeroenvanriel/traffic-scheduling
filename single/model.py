import numpy as np
import torch
import torch.nn as nn

class ActionTransform(nn.Module):

    def inverse_action_transform(self, automaton, lane):
        # stay on last_lane when action == 0
        # switch to other lane when action == 1
        if automaton.last_lane == None:
            # assume initial last_lane == 0
            out = lane
        else:
            out = int(automaton.last_lane != lane)

        return torch.as_tensor(out, dtype=torch.float, device=torch.device('cuda'))

    def action_transform(self, automaton, logit):
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

    def heuristic(self, automaton):
        """Perform the actual trained heuristic."""
        return self.action_transform(automaton, self.forward(self.state_transform(automaton)))


class PaddedEmbeddingModel(ActionTransform):
    """Should be used with 2 lanes, because of how actions are transformed to lanes."""

    def __init__(self, lanes, horizon):
        super().__init__()
        self.horizon = horizon
        self.lanes = lanes
        self.network = nn.Sequential(
            nn.Linear(lanes * horizon, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x)

    def state_transform(self, automaton):
        # compute minimum LB of unscheduled vehicles
        LBs = []
        for LB_lane, k_lane in zip(automaton.LB, automaton.k):
            LBs.extend(LB_lane[k_lane:])
        min_LB = min(LBs)

        # create the observation padded with zeros
        obs = np.zeros((self.lanes, self.horizon))
        for l, (LB_lane, k_lane) in enumerate(zip(automaton.LB, automaton.k)):
            actual_horizon = min(self.horizon, len(LB_lane) - k_lane)
            obs[l,:actual_horizon] = LB_lane[k_lane:k_lane+actual_horizon] - min_LB

        # lane cycling
        last_lane = automaton.last_lane or 0
        obs = np.roll(obs, last_lane, axis=0)

        return torch.as_tensor(obs.flatten(), dtype=torch.float, device=torch.device('cuda'))


class RecurrentEmbeddingModel(ActionTransform):

    def __init__(self, lanes, max_horizon):
        super().__init__()
        self.lanes = lanes
        self.max_horizon = max_horizon

        self.rnn_out = 32
        self.rnn = nn.RNN(1, self.rnn_out)
        self.network = nn.Sequential(
            nn.Linear(self.lanes * self.rnn_out, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, obs):
        N = obs.size()[0] # batch size

        # take ragged tensor of horizons and apply rnn for each lane
        embedding = torch.zeros((N, self.lanes, self.rnn_out))

        for n in range(N):
            for l in range(self.lanes):
                length = int(obs[n, l, 0].item())
                if length == 0:
                    continue

                inp = torch.unsqueeze(obs[n, l,:length], 1)
                out, _ = self.rnn(inp)
                # we take the last output as embedding
                embedding[n, l] = out[-1]

        return self.network(torch.flatten(embedding, 1, 2).cuda())

    def state_transform(self, automaton):
        # compute minimum LB of unscheduled vehicles
        LBs = []
        for LB_lane, k_lane in zip(automaton.LB, automaton.k):
            LBs.extend(LB_lane[k_lane:])
        min_LB = min(LBs)

        # ragged array; first number indicates length of the sequence
        obs = np.zeros((self.lanes, self.max_horizon))
        lengths = np.empty((self.lanes, 1))
        for l, (LB_lane, k_lane) in enumerate(zip(automaton.LB, automaton.k)):
            actual_horizon = min(len(LB_lane) - k_lane, self.max_horizon)
            lengths[l] = actual_horizon
            obs[l, :actual_horizon] = LB_lane[k_lane:k_lane + actual_horizon] - min_LB

        out = np.hstack([lengths, obs])

        # lane cycling
        last_lane = automaton.last_lane or 0
        out = np.roll(out, last_lane, axis=0)
        out = np.expand_dims(out, axis=0) # to allow stacking with actions

        return torch.as_tensor(out, dtype=torch.float, device=torch.device('cuda'))
