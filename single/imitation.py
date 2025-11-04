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
#
# > ⚠️ **Warning:** This notebook file is paired to a .py file with the same name, such that we can cleanly import the functionality from other notebooks. This is done using the facilities of the jupytext package. The cells in this notebook that are only meant as "demonstration" are marked with the cell tag "active-ipynb", which causes the jupytext synchronization command to ignore these when syncing to the .py file. This is our current way of doing "literate programming" with jupyter notebooks.

# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# ## Imitation Learning

# %% [markdown]
# Optional further refinements:
# - cache solved train instances across multiple instances of ImitationLearning class
# - separately measure time needed to solve train instances

# %%
import torch
import torch.nn as nn

class RecurrentEmbeddingModel(nn.Module):

    def __init__(self, R):
        super().__init__()
        self.R = R

        self.rnn_out = 32
        self.rnn = nn.RNN(1, self.rnn_out)
        self.network = nn.Sequential(
            nn.Linear(self.R * self.rnn_out, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, h, l):
        N = h.size()[0] # batch size

        # take ragged tensor of horizons and apply rnn for each lane
        embedding = torch.zeros((N, self.R, self.rnn_out))

        for n in range(N):
            for r in range(self.R):
                # first number indicates length of the horizon
                length = int(l[n, r, 0].item())
                if length == 0:
                    continue

                # rest of sequence is actual horizon
                horizon = h[n, r, :length]
                # reverse ("flip up down") the horizon
                inp = torch.flipud(horizon)
                # add dimension as required by RNN
                out, _ = self.rnn(torch.unsqueeze(inp, 1))
                # take the output at the last step as embedding
                embedding[n, r] = out[-1]

        return self.network(torch.flatten(embedding, 1, 2)).flatten()


# %%
from single.mdp import SingleScheduleEnv, HorizonObservationWrapper, HorizonRollingWrapper

import torch
import torch.optim as optim
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.utils.data import TensorDataset, DataLoader

import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

class ImitationLearning:

    def __init__(self, max_k=5):
        self.max_k = max_k

    def train(self, gen, name=None, **options):
        # 1. obtain expert data
        # this involves computing near-optimal solutions by solving MILPs,
        # which becomes very expensive for larger instances
        # (so we might share this step, if we want to test multiple 
        #  configurations of imitation learning)
        N_train = 20
        states, actions = self._generate_expert_demonstration(gen, N_train, name)

        # 2. fit model to expert data state-action pairs
        total_steps = 2000
        batch_size = 20
        self.model = RecurrentEmbeddingModel(2)

        all_horizons, all_lengths = [], []
        for d in states:
            horizon_tensor = torch.tensor(d['horizon'], dtype=torch.float32)
            lengths_tensor = torch.tensor(d['h_lengths'], dtype=torch.long).unsqueeze(dim=1)
            all_horizons.append(horizon_tensor)
            all_lengths.append(lengths_tensor)

        train_set = TensorDataset(
            torch.stack(all_horizons, dim=0),
            torch.stack(all_lengths, dim=0),
            torch.tensor(actions, dtype=torch.float32)
        )
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

        learning_rate = 1e-4
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        train_losses = []
        step = 0
        with tqdm(total=total_steps, desc=f"training", leave=False) as pbar:
            while True:
                for h, l, a in train_loader:
                    # perform a single training step
                    self.model.train()
                    optimizer.zero_grad()
                    action = self.model(h, l)
                    loss = binary_cross_entropy_with_logits(action, a)
                    loss.backward()
                    optimizer.step()
                    train_losses.append(loss.item())

                    step += 1
                    pbar.update(1)

                    if step >= total_steps: break # ...out of inner loop
                if step >= total_steps: break # ...out of outer loop
        
        self.train_losses = train_losses
        return train_losses

    def report(self, name):
        filename = f"results/cache/imitation_losses_{name}.pkl"
        with open(filename, "wb") as file:
            pickle.dump(self.train_losses, file)
        plt.plot(self.train_losses)
        plt.xlabel(r'timestep')
        plt.ylabel(r'loss')
        plt.savefig(f'results/figures/imitation_losses_{name}.pdf')
        plt.close()

    def eval(self, s):
        env = SingleScheduleEnv(instance=s)
        env = HorizonObservationWrapper(env, max_vehicles_per_route=self.max_k)
        env = HorizonRollingWrapper(env)
        obs, _ = env.reset()
        done = False
        total_reward = 0
        self.model.eval()
        while not done: 
            logit = self.model(
                torch.tensor(obs['horizon'], dtype=torch.float32).unsqueeze(0),
                torch.tensor(obs['h_lengths'], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            )
            action = logit > 0
            obs, reward, done, *_ = env.step(action)
            total_reward += reward
        return -total_reward


    def _generate_expert_demonstration(self, gen, N, name):
        filename = f"results/cache/expert-demo-{name}.pkl"

        path = Path(filename)
        if path.is_file():
            with open(filename, "rb") as f:
                states, actions = pickle.load(f)
            print("Loading expert demonstration from cache")
        else:
            states, actions = [], []
            for _ in tqdm(range(N), desc="Generating expert demonstration"):
                # 1a. generate training instance
                s = gen()

                # 1b. compute optimal solution
                s.solve(timelimit=60)

                # 1c. compute route order ("eta")
                route_order = s.opt.route_order

                # 1c. play optimal route order on the MDP to extract expert demonstration
                env = SingleScheduleEnv(instance_generator=lambda: s)
                env = HorizonObservationWrapper(env, max_vehicles_per_route=self.max_k)
                env_cycling = HorizonRollingWrapper(env)

                obs, _ = env.reset()
                state = env_cycling._observation(obs)
                done = False
                eta = iter(route_order) # replay optimal route order
                while not done:
                    route = next(eta)

                    # record current state + action
                    # (so note that we won't record the final state of this episode)
                    states.append(state)
                    actions.append(env_cycling._inverse_action(route))

                    obs, _, done, _, _ = env.step(route)
                    state = env_cycling._observation(obs)
            
            with open(filename, "wb") as file:
                pickle.dump([states, actions], file)

        return states, actions

# %% [markdown]
# ## Demo

# %% tags=["active-ipynb"]
# from single.basics import bimodal_exponential, generate_instance
# F = bimodal_exponential(p=0.1, s1=0.1, s2= 5.6)
# gen = lambda: generate_instance(F, n=[20, 20, 20])

# %% tags=["active-ipynb"]
# im = ImitationLearning()
# losses = im.train(gen)

# %% tags=["active-ipynb"]
# N = 100 
# total = 0
# instances = []
# for _ in tqdm(range(N), desc="Generating/solving train instances"):
#     # 1a. generate training instance
#     s = gen()
#
#     # 1b. compute optimal solution
#     s.solve(consolelog=False)
#     instances.append(s)
#
#     total += s.opt.delay
# print(total / N)

# %% tags=["active-ipynb"]
# score = 0
# for s in instances:
#     score += im.eval(s)
# print(score / len(instances))

# %% tags=["active-ipynb"]
# import matplotlib.pyplot as plt
# plt.plot(losses)

# %% tags=["active-ipynb"]
# N = 100
# score = 0
# for _ in range(N):
#     score += im.eval(gen())
# print(score / N)

# %% tags=["active-ipynb"]
# from single.mdp import SingleScheduleEnv, HorizonObservationWrapper, HorizonRollingWrapper, current_route
# from single.basics import generate_simple_instance
#
# options = { 'render_roll': False }
# instance = generate_simple_instance(n=[3,3])
# rho = instance.rho
#
# env = SingleScheduleEnv(instance=instance, options=options)
# env = HorizonObservationWrapper(env)
# roll_env = HorizonRollingWrapper(env)
#
# route_order = [0, 1, 1, 0, 0, 1]
# rolled_actions = []
#
# env.reset()
# print('current route=', current_route(env.unwrapped))
# env.render()
# for r in route_order:
#     rolled_actions.append(roll_env._inverse_action(r))
#     env.step(r)
# env.render()

# %% tags=["active-ipynb"]
# env.reset()
# env.render()
#
# for r_roll in rolled_actions:
#     roll_env.step(r_roll)
#
# env.render()

# %% [markdown]
# ### RNN model

# %% tags=["active-ipynb"]
# import torch
# import torch.nn as nn
# from torch.nn.utils.rnn import pack_padded_sequence
#
# horizon = torch.tensor([
#     [0.5, 1.7, 2.9],
#     [0.0, 2.1731577, 0.0]
# ], dtype=torch.float32)
#
# h_lengths = torch.tensor([3, 2], dtype=torch.long)
#
# # add feature dimension (since each element is scalar)
# horizon = horizon.unsqueeze(-1)  # shape [batch, seq_len, input_size] = [2, 3, 1]
#
# # define encoder
# class RNNEncoder(nn.Module):
#     def __init__(self, hidden_size, output_size):
#         super().__init__()
#         self.rnn = nn.GRU(1, hidden_size, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)
#
#     def forward(self, padded, lengths):
#         # pack padded sequences to ignore zero padding
#         packed = pack_padded_sequence(padded, lengths.cpu(), batch_first=True, enforce_sorted=False)
#         _, hidden = self.rnn(packed)
#         hidden = hidden[-1]  # last layer hidden state
#         encoded = self.fc(hidden)
#         return encoded
#
# encoder = RNNEncoder(hidden_size=8, output_size=4)
# encoded = encoder(horizon, h_lengths)
#
# print("Encoded shape:", encoded.shape)
# print("Encoded output:\n", encoded)
#
