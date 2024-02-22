import numpy as np
import gymnasium as gym

import single_intersection_gym

### generate schedule from simple policy
### TODO: generalize

def evaluate(env, policy):
    obs, _ = env.reset()
    total_reward = 0
    terminated = False
    while not terminated:
        action = policy(obs)
        next_obs, reward, terminated, _, _ = env.step(action)
        total_reward += reward
        obs = next_obs

    return total_reward


# load instance
instance_file = snakemake.input[0]
instance = np.load(instance_file)

# apply simple policy in gym environment
def policy(obs):
    return 0
# horizon does not matter for the current policy
horizon = 0
env = gym.make("SingleIntersectionEnv", K=instance['K'], instance_generator=lambda: instance,
                switch_over=instance['s'], horizon=horizon)
obj = evaluate(env, policy)

# save to file
np.savez(snakemake.output[0], obj=obj)
