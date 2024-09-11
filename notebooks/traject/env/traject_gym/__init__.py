from gymnasium.envs.registration import register
from .env import TrajectEnv

register(
     id="TrajectEnv",
     entry_point="traject_gym:TrajectEnv",
     max_episode_steps=300,
)
