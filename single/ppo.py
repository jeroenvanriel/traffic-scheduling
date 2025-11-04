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
# ## Proximal Policy Optimization (PPO)

# %%
from single.mdp import SingleScheduleEnv, HorizonObservationWrapper, HorizonRollingWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor
import matplotlib.pyplot as plt
import pandas as pd
from util import TqdmCallback
import os

os.makedirs("results/ppologs/", exist_ok=True)

class PpoRl:
    def __init__(self, steps=100, max_k=5, verbose=1):
        self.steps = steps
        self.max_k = 5
        self.verbose = verbose

    def make_wrapped_env(self, gen=None, instance=None):
        env = SingleScheduleEnv(instance_generator=gen, instance=instance)
        env = HorizonObservationWrapper(env, max_vehicles_per_route=self.max_k)
        env = HorizonRollingWrapper(env)
        return env

    def train(self, gen, name):
        self.gen = gen
        # just inspect a sample to determine model feature "size"
        s = gen()
        self.R = s.R

        vec_env = make_vec_env(lambda: self.make_wrapped_env(gen=gen), n_envs=4)
        vec_env = VecMonitor(vec_env, filename=f"results/ppologs/{name}")
        self.model = PPO("MultiInputPolicy", vec_env,
                         device='cpu', verbose=self.verbose) \
                            .learn(total_timesteps=self.steps,
                                   callback=TqdmCallback(self.steps))

    def eval(self, instance, gen=None):
        env = self.make_wrapped_env(instance=instance, gen=gen)
        env = Monitor(env)

        # Evaluate the policy
        mean_reward, std_reward = evaluate_policy(
            self.model, env,
            n_eval_episodes=1,     # number of episodes to test
            render=False,            # set True to see rendering
            deterministic=True,      # remove exploration noise
            return_episode_rewards=False
        )

        # print(f"Mean reward: {mean_reward} +/- {std_reward}")
        return -mean_reward
    
    def report(self, name):
        df = pd.read_csv(f"results/ppologs/{name}.monitor.csv", comment="#")

        window = 50
        df["r_smooth"] = df["r"].rolling(window, min_periods=1).mean()

        plt.figure(figsize=(8,4))
        plt.xlabel("Time (s)")
        plt.ylabel("Episode reward")
        plt.plot(df["t"], df["r"], alpha=0.3, label="Episodic reward")
        plt.plot(df["t"], df["r_smooth"], linewidth=2, label=f"Smoothed ({window})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'results/figures/ppo_{name}.pdf')
        plt.close()

# %% [markdown]
# ## Demo
#
# > ⚠️ **Warning:** Remember to tag these cells as `active-ipynb`.

# %% [markdown]
# ### Plot demo

# %% tags=["active-ipynb"]
# from single.basics import generate_instance, bimodal_exponential
# from single.mdp import draw_horizon_obs
#
# F = bimodal_exponential(p=0.1, s1=0.1, s2=5.6)
# gen = lambda: generate_instance(F, n=[2, 2])
# rho = gen().rho
#
# env = SingleScheduleEnv(instance_generator=gen, options={'render_roll': False})
# env = HorizonObservationWrapper(env)
# # env = HorizonRollingWrapper(env)
# obs, _ = env.reset()
# env.render()
# draw_horizon_obs(obs, rho)
# obs, *_ = env.step(0)
# draw_horizon_obs(obs, rho)

# %% [markdown]
# ### Learning demo

# %% tags=["active-ipynb"]
# from single.basics import generate_instance, uniform
#
# model = PpoRl(steps=200_000, verbose=1)
#
# F = uniform()
# gen = lambda: generate_instance(F, n=[20, 20, 20])
# model.train(gen, "test")
# score = model.eval(None, gen=gen)

# %% tags=["active-ipynb"]
# from stable_baselines3.common.results_plotter import load_results
#
# %cd /home/jeroen/repos/traffic-scheduling/single
# df = load_results("./logs")
# print(df.tail(10))
# import matplotlib.pyplot as plt
#
# window = 50
# df["r_smooth"] = df["r"].rolling(window, min_periods=1).mean()
#
# plt.figure(figsize=(8,4))
# plt.plot(df["t"], df["r"], alpha=0.3, label="Episodic reward")
# plt.plot(df["t"], df["r_smooth"], linewidth=2, label=f"Smoothed ({window})")
# plt.xlabel("Time (s)")
# plt.ylabel("Episode reward")
# plt.legend()
# plt.grid(True)
# plt.show()

# %% tags=["active-ipynb"]
# env = SingleScheduleEnv(instance_generator=gen, instance=None)
# env = HorizonObservationWrapper(env, max_vehicles_per_route=5)
# env = HorizonRollingWrapper(env)
# env = Monitor(env)
#
# # Evaluate the policy
# mean_reward, std_reward = evaluate_policy(
#     model.model, env,
#     n_eval_episodes=100,     # number of episodes to test
#     render=False,            # set True to see rendering
#     deterministic=True,      # remove exploration noise
#     return_episode_rewards=False
# )
#
# print(f"Mean reward: {mean_reward} +/- {std_reward}")

# %% tags=["active-ipynb"]
# F = uniform()
# gen = lambda: generate_instance(F, n=[2, 2, 2])
#
# env = SingleScheduleEnv(instance_generator=gen, instance=None)
# env = HorizonObservationWrapper(env, max_vehicles_per_route=2)
# env = HorizonRollingWrapper(env)
#
# rho = gen().rho
#
# from single.mdp import draw_horizon_obs
#
# obs, _ = env.reset()
# draw_horizon_obs(obs, rho)
#
# obs, *_ = env.step(0)
# draw_horizon_obs(obs, rho)
#
# obs, *_ = env.step(0)
# draw_horizon_obs(obs, rho)
#

# %%
