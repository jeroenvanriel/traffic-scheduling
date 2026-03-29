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
from traffic_scheduling.single.mdp import SingleScheduleEnv, NewHorizonObservationWrapper, NewHorizonRollingWrapper, PaddedHorizonWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import ts2xy, window_func
import matplotlib.pyplot as plt
import torch as th
import numpy as np
from traffic_scheduling.single.util import TqdmCallback
import os

os.makedirs("results/ppologs/", exist_ok=True)


# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# ## Proximal Policy Optimization (PPO)

# %%
class PpoRl:
    def __init__(self, max_k=5, steps=100, n_envs=4, verbose=0):
        self.max_k = max_k
        self.steps = steps
        self.n_envs = n_envs
        self.verbose = verbose

    def make_wrapped_env(self, instance=None, gen=None):
        env = SingleScheduleEnv(instance=instance, instance_generator=gen)
        env = NewHorizonObservationWrapper(env, gaps=False)
        env = NewHorizonRollingWrapper(env)
        env = PaddedHorizonWrapper(env, max_k=self.max_k, reversed=False)
        return env

    def train(self, gen, name):
        vec_env = make_vec_env(lambda: self.make_wrapped_env(gen=gen),
                               n_envs=self.n_envs, monitor_dir=f"results/ppologs/{name}")
        policy_kwargs = dict(activation_fn=th.nn.ReLU)
        self.model = PPO("MultiInputPolicy", vec_env, policy_kwargs=policy_kwargs, device='cpu', verbose=self.verbose) \
                            .learn(total_timesteps=self.steps, callback=TqdmCallback(self.steps, n_envs=self.n_envs))

    def eval(self, instance):
        env = Monitor(self.make_wrapped_env(instance=instance))
        mean_reward, _ = evaluate_policy(self.model, env, n_eval_episodes=1)
        return -mean_reward
    
    def report(self, name):
        df = load_results(f"results/ppologs/{name}")

        window = 50
        df["r_smooth"] = df["r"].rolling(window, min_periods=1).mean()

        # Convert dataframe (x=timesteps, y=episodic return)
        x, y = ts2xy(df, "timesteps")

        # plot raw data
        plt.figure(figsize=(8, 6))
        plt.subplot(2, 1, 1)
        plt.scatter(x, y, s=2, alpha=0.6)
        plt.xlabel("Timesteps")
        plt.ylabel("Episode Reward")
        plt.title("Raw Episode Rewards")

        # plot smoothed data with custom window
        plt.subplot(2, 1, 2)
        window = 50
        if len(x) >= window:  # Only smooth if we have enough data
            x_smooth, y_smooth = window_func(x, y, 50, np.mean)
            plt.plot(x_smooth, y_smooth, linewidth=2)
            plt.xlabel("Timesteps")
            plt.ylabel("Average Episode Reward")
            plt.title(f"Smoothed Episode Rewards (window size  = {window})")

        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'results/figures/ppo_{name}.pdf')
        plt.close()

# %% tags=["active-ipynb"]
# from single.basics import bimodal_exponential, generate_instance
#
# F = bimodal_exponential(p=0.1, s1=0.1, s2= 5.6)
# gen = lambda: generate_instance(F, n=[5, 5, 5])
#
# ppo = PpoRl(steps=100_000)
# ppo.train(gen, 'test')
#
# N = 100
# sum([ppo.eval(gen()) for _ in range(N)]) / N
# ppo.report('test')
