from gymnasium.envs.registration import register
from .env import SingleIntersectionEnv

register(
     id="SingleIntersectionEnv",
     entry_point="single_intersection_gym:SingleIntersectionEnv",
     max_episode_steps=300,
)
