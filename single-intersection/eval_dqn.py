import numpy as np
import torch
import gymnasium as gym

import single_intersection_gym


def evaluate(env, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    obs, info = env.reset()
    total_reward = info['initial_reward']
    terminated = False
    while not terminated:
        q_values = model(torch.Tensor(obs).to(device))
        action = torch.argmax(q_values).cpu().numpy()
        next_obs, reward, terminated, _, info = env.step(action)
        total_reward += reward
        obs = next_obs

    return total_reward, info


# horizon
horizon = snakemake.params['horizon']

model_path = snakemake.input[0]
model = torch.load(model_path)

for i, instance_file in enumerate(snakemake.input[1:]):
    instance = np.load(instance_file)
    K = instance['K']

    env = gym.make("SingleIntersectionEnv", K=instance['K'], instance=instance,
                   switch_over=instance['s'], horizon=horizon)
    env = gym.wrappers.FlattenObservation(env)

    obj, info = evaluate(env, model)

    schedule = {
        **{f"start_time_{i}": info['start_time'][i] for i in range(K)},
        **{f"end_time_{i}": info['end_time'][i] for i in range(K)},
    }

    np.savez(snakemake.output[i], obj=obj, K=K, **schedule)
