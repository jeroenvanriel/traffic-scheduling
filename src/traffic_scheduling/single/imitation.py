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
#
# > ⚠️ **Warning:** This notebook file is paired to a .py file with the same name, such that we can cleanly import the functionality from other notebooks. This is done using the facilities of the jupytext package. The cells in this notebook that are only meant as "demonstration" are marked with the cell tag "active-ipynb", which causes the jupytext synchronization command to ignore these when syncing to the .py file. This is our current way of doing "literate programming" with jupyter notebooks.

# %%
from traffic_scheduling.single.mdp import SingleScheduleEnv, HorizonObservationWrapper, HorizonRollingWrapper

import torch as th
import torch.nn as nn

import torch.optim as optim
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy
from torch.utils.data import TensorDataset, DataLoader

import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm


# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# ## Imitation Learning

# %% [markdown]
# ### Generate expert demonstration

# %% [markdown]
# This involves computing near-optimal solutions by solving MILPs, which becomes very expensive for larger instances
# (so we cache this step, because we want to test multiple configurations of imitation learning, but with the same problem distribution).

# %%
def generate_expert_demonstration(gen, name, N, timelimit, max_k):
    filename = f"results/cache/expert-train-schedules-{name}.pkl"

    if Path(filename).is_file():
        with open(filename, "rb") as f:
            instances = pickle.load(f)
        print("Loading optimal schedules for expert demonstration from cache")
        solve = False
    else:
        instances = []
        solve = True

    states, actions = [], []
    for i in tqdm(range(N), desc="Generating expert demonstration"):
        if solve:
            # generate training instance and compute optimal solution
            s = gen()
            s.solve(cutting_planes=[2,3], timelimit=timelimit)
            instances.append(s)
        else:
            # get cached instance that is already solved 
            s = instances[i]

        # 1c. compute route order ("eta")
        route_order = s.opt.route_order

        # 1c. play optimal route order on the MDP to extract expert demonstration
        env = SingleScheduleEnv(instance=s)
        env = HorizonObservationWrapper(env, max_k=max_k)
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
    
    if solve:
        with open(filename, "wb") as file:
            pickle.dump(instances, file)

    return states, actions


# %% [markdown]
# ### Recurrent horizon embedding model

# %%
class RecurrentEmbeddingModel(nn.Module):

    def __init__(self, R):
        super().__init__()
        self.R = R

        net_out = 1 if R == 2 else R
        self.rnn_out = 32
        self.rnn = nn.RNN(1, self.rnn_out)
        self.network = nn.Sequential(
            nn.Linear(self.R * self.rnn_out, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, net_out),
        )

    def forward(self, h, l):
        N = h.size()[0] # batch size

        # take ragged tensor of horizons and apply rnn for each lane
        embedding = th.zeros((N, self.R, self.rnn_out))

        for n in range(N):
            for r in range(self.R):
                # first number indicates length of the horizon
                length = int(l[n, r, 0].item())
                if length == 0:
                    continue

                # rest of sequence is actual horizon
                horizon = h[n, r, :length]
                # reverse ("flip up down") the horizon
                inp = th.flipud(horizon)
                # add dimension as required by RNN
                out, _ = self.rnn(th.unsqueeze(inp, 1))
                # take the output at the last step as embedding
                embedding[n, r] = out[-1]

        logits = self.network(th.flatten(embedding, 1, 2))
        if self.R == 2: logits = logits.flatten() # binary logits is 1d
        return logits


# %% [markdown]
# ### Imitation learning

# %%
from dataclasses import dataclass

@dataclass
class ImitationLearning:
    N_train: int = 20
    timelimit: int = 20
    max_k: int = 5
    total_steps: int = 2000
    batch_size: int = 20
    learning_rate: float = 1e-4

    def train(self, gen, name='test', **options):
        # 1. prepare expert demonstration
        # (state-action pairs leading to optimal schedules)
        states, actions = generate_expert_demonstration(gen, name, self.N_train, self.timelimit, self.max_k)

        all_horizons = [th.tensor(d['horizon'], dtype=th.float32) for d in states]
        all_lengths = [th.tensor(d['h_lengths'], dtype=th.long).unsqueeze(dim=1) for d in states]

        train_set = TensorDataset(
            th.stack(all_horizons, dim=0),
            th.stack(all_lengths,  dim=0),
            th.tensor(actions, dtype=th.float32)
        )
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)

        # 2. initialize model and loss function
        self.R = gen().R # just sample an instance to get the number of routes
        self.model = RecurrentEmbeddingModel(self.R)
        loss_func = binary_cross_entropy_with_logits if self.R == 2 else lambda input, target: cross_entropy(input, target.long())

        # 3. fit model to expert data state-action pairs
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.train_losses = []
        step = 0
        with tqdm(total=self.total_steps, desc=f"training", leave=False) as pbar:
            while True:
                for h, l, a in train_loader:
                    # perform a single training step
                    self.model.train()
                    optimizer.zero_grad()
                    action = self.model(h, l)
                    loss = loss_func(action, a)
                    loss.backward()
                    optimizer.step()
                    self.train_losses.append(loss.item())

                    step += 1
                    pbar.update(1)

                    if step >= self.total_steps: break # ...out of inner loop
                if step >= self.total_steps: break # ...out of outer loop

    def report(self, name=None):
        if name is not None:
            filename = f"results/cache/imitation_losses_{name}.pkl"
            with open(filename, "wb") as file:
                pickle.dump(self.train_losses, file)
        plt.plot(self.train_losses)
        plt.xlabel(r'timestep')
        plt.ylabel(r'loss')
        if name is not None:
            plt.savefig(f'results/figures/imitation_losses_{name}.pdf')
        else:
            plt.show()
        plt.close()

    def eval(self, s):
        env = SingleScheduleEnv(instance=s)
        env = HorizonObservationWrapper(env, max_k=self.max_k)
        env = HorizonRollingWrapper(env)
        obs, _ = env.reset()
        done = False
        total_reward = 0
        self.model.eval()
        while not done: 
            logit = self.model(
                th.tensor(obs['horizon'], dtype=th.float32).unsqueeze(0),
                th.tensor(obs['h_lengths'], dtype=th.long).unsqueeze(0).unsqueeze(-1)
            )
            if self.R == 2:
                action = logit > 0
            else:
                action = th.argmax(logit)
            obs, reward, done, *_ = env.step(action)
            total_reward += reward
        return -total_reward


# %% [markdown]
# ## Verification and experiments

# %% [markdown]
# ### Verify action playback

# %% [markdown]
# Verify that replaying the actions on the environment yields again the original schedule.

# %% tags=["active-ipynb"]
# from single.basics import bimodal_exponential, generate_instance
#
# F = bimodal_exponential(p=0.1, s1=0.1, s2= 5.6)
# n = [5, 5, 5]
# name = 'verify_' + '_'.join(str(nr) for nr in n)
# gen = lambda: generate_instance(F, n=n)
#
# states, actions = generate_expert_demonstration(gen, name, 1, timelimit=10, max_k=5)
#
# filename = f"results/cache/expert-train-schedules-{name}.pkl"
# with open(filename, "rb") as f:
#     instances = pickle.load(f)
# instance = instances[0]
#
# env = SingleScheduleEnv(instance=instance)
# env = HorizonObservationWrapper(env, max_k=5)
# env = HorizonRollingWrapper(env)
#
# env.reset()
# for a in actions:
#     env.step(a)
#
# from numpy import allclose
# assert allclose(env.unwrapped.LB, instance.opt.y)

# %% [markdown]
# ### Measure learning performance

# %% tags=["active-ipynb"]
# from single.basics import bimodal_exponential, generate_instance
#
# F = bimodal_exponential(p=0.1, s1=0.1, s2= 5.6)
# n = [5, 5, 5]
# name = 'experiment_' + '_'.join(str(nr) for nr in n)
# gen = lambda: generate_instance(F, n=n)
#
# N_train = 40
# max_k = 5
# states, actions = generate_expert_demonstration(gen, name, N_train, timelimit=10, max_k=max_k)

# %% tags=["active-ipynb"]
# im = ImitationLearning(batch_size=10)
# losses = im.train(gen, name)
# im.report()

# %% tags=["active-ipynb"]
# def generate_test_set(N_test=100, name='exp'):
#     filename = f"results/cache/expert-test-schedules-{name}.pkl"
#
#     if Path(filename).is_file():
#         with open(filename, "rb") as f:
#             instances, opt_delay = pickle.load(f)
#         print("Loading optimal schedules for expert demonstration from cache")
#     else:
#         total_delay = 0
#         instances = []
#         for _ in tqdm(range(N_test), desc="Generating/solving test instances"):
#             # 1a. generate training instance
#             s = gen()
#
#             # 1b. compute optimal solution
#             s.solve(cutting_planes=[2,3])
#             instances.append(s)
#
#             # 1c. keep track of total delay (for average)
#             total_delay += s.opt.delay
#
#         opt_delay = total_delay / N_test
#
#         with open(filename, "wb") as file:
#             pickle.dump((instances, opt_delay), file)
#
#     return instances, opt_delay
#
# test_instances, opt_obj = generate_test_set(100, name)
#
# im_delay = 0
# for s in test_instances: im_delay += im.eval(s)
# im_obj = im_delay / len(test_instances)
#
# print(f"{'MILP obj':<15} = {opt_obj}")
# print(f"{'imitation obj':<15} = {im_obj}")
# print(f"{'gap':<15} = {(im_obj - opt_obj) / opt_obj * 100:.2f}%")

# %%
