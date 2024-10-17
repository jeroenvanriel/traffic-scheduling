import numpy as np
from tqdm import tqdm, trange

import pickle

import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import single_intersection_gym

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from exact import solve
from expert_demonstration import expert_demonstration
from plotting import plot_instance, plot_schedule

from captum.attr import IntegratedGradients


# generate training data
seed=31307741687469044381975587942973893579
rng = np.random.default_rng(seed)


def lane(spec):
    """Generate a stream of arrivals for a single lane."""

    length = rng.integers(1, spec['theta'], size=(spec['n']))
    length_shifted = np.roll(length, 1)
    length_shifted[0] = 0

    interarrival = rng.exponential(scale=spec['lambda'], size=(spec['n']))
    arrival = np.cumsum(interarrival + length_shifted)

    return arrival, length


def generate_instance(spec):
    arrival0, length0 = lane(spec)
    arrival1, length1 = lane(spec)

    return {
        'arrival0': arrival0, 'length0': length0,
        'arrival1': arrival1, 'length1': length1,
        **spec
    }


def evaluate_schedule(schedule):
    total_reward = 0

    for k in range(schedule['K']):
        for arrival, length, start in zip(
                schedule[f'arrival{k}'], schedule[f'length{k}'], schedule[f'start_time_{k}']
        ):
            # minus sign because penalty
            total_reward -= length * (start - arrival)

    return total_reward


spec = { 'K': 2, 'n': 10, 's': 2, 'horizon': 4, 'theta': 2, 'lambda': 2 }

# generate training data set
train_N = 4000
train_states = []
train_actions = []
try:
    with open('train.pkl', 'rb') as file:
        print('loading train data')
        train_states, train_actions = pickle.load(file)
except FileNotFoundError:
    print('generating train data')
    for _ in trange(train_N):
        # get one instance
        instance = generate_instance(spec)

        # solve this instance
        schedule = solve(instance)

        # generate expert demonstration
        states, actions = expert_demonstration(instance, schedule)

        train_states.extend(states)
        train_actions.extend(actions)

    # creating single numpy arrays
    train_states = np.vstack(train_states)
    train_actions = np.vstack(train_actions)

    with open('train.pkl', 'wb') as file:
        pickle.dump((train_states, train_actions), file)

train_states = torch.as_tensor(train_states, dtype=torch.float, device=torch.device('cuda'))
train_actions = torch.as_tensor(train_actions, dtype=torch.float, device=torch.device('cuda'))
train_set = TensorDataset(train_states, train_actions)


# generate test data set
test_N = 100
test_set = []
test_obj = 0 # mean test obj
try:
    with open('test.pkl', 'rb') as file:
        print('loading test data')
        test_set, test_obj = pickle.load(file)
except FileNotFoundError:
    print('generating test data')
    for i in trange(test_N):
        spec['n'] = 20                                      # <-- !!!
        instance = generate_instance(spec)
        schedule = solve(instance)
        test_set.append(schedule)
        plot_schedule(schedule, out=f'{i}_exact.pdf')
        test_obj += schedule['obj']

    test_obj = test_obj / test_N
    with open('test.pkl', 'wb') as file:
        pickle.dump((test_set, test_obj), file)


def test(model, plot=False):
    model.eval()
    print('--- test ---')
    model_obj = 0
    for i, instance in enumerate(test_set):
        env = gym.make("SingleIntersectionEnv", instance=instance)
        env = FlattenObservation(env)
        obs, info = env.reset()
        terminated = False
        while not terminated:
            obs = torch.tensor(obs).cuda()
            logit = model(obs)
            a = logit > 0
            a = int(a.detach().cpu().numpy()[0])
            obs, reward, terminated, _, info = env.step(a)

        instance['start_time_0'] = info['start_time'][0]
        instance['start_time_1'] = info['start_time'][1]
        obj = evaluate_schedule(instance)
        if plot:
            plot_instance(instance, out=f'{i}_instance.pdf')
            plot_schedule(instance, out=f'{i}_model.pdf')
            print(i, obj, instance['obj'], abs(obj - instance['obj']) < 0.01)
        model_obj += obj

    model_obj = model_obj / test_N

    # report test score
    print(f'(model_obj = {model_obj}) / (test_obj = {test_obj}) = {model_obj / test_obj}')


class Model(nn.Module):
    def __init__(self, in_shape):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(in_shape).prod(), 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x)

in_shape = train_states[0].shape
model = Model(in_shape)
model.cuda()

test(model)

# train
epochs = 10
batch_size = 10
data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

model.train()
print('--- train ---')
for i in range(epochs):
    loss_total = 0
    print(f'epoch: {i}')
    for s, a in tqdm(data_loader):
        optimizer.zero_grad()
        logit = model(s)
        loss = F.binary_cross_entropy_with_logits(logit, a)
        loss.backward()
        loss_total += loss
        optimizer.step()
    print(loss_total)

test(model, plot=True)


integrated_gradients = IntegratedGradients(model)

x = train_states[0:10]
print(x.shape)
attributions_ig = integrated_gradients.attribute(x, n_steps=200)

print(attributions_ig)
