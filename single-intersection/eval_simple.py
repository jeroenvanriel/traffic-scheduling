import numpy as np
import gymnasium as gym

import single_intersection_gym

### generate schedule from simple policy
### TODO: generalize this procedure for different simple policies

def evaluate(env, policy):
    obs, info = env.reset()
    total_reward = info["initial_reward"]
    terminated = False
    while not terminated:
        action = policy(obs)
        next_obs, reward, terminated, _, info = env.step(action)
        total_reward += reward
        obs = next_obs

    return total_reward, info


# load instance
instance_file = snakemake.input[0]
instance = np.load(instance_file)
K = instance['K']

# apply simple policy in gym environment
def policy(obs):
    return 1 # switch always (exhaustive policy)

horizon = 10

env = gym.make("SingleIntersectionEnv", K=K, instance_generator=lambda: instance,
                switch_over=instance['s'], horizon=horizon)

obj, info = evaluate(env, policy)

schedule = {
    **{f"start_time_{i}": info['start_time'][i] for i in range(K)},
    **{f"end_time_{i}": info['end_time'][i] for i in range(K)},
}

# save to file
np.savez(snakemake.output[0], obj=obj, K=K, **schedule)
