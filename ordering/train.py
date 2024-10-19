import numpy as np
import pickle
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from automaton import Automaton, evaluate
from model import PaddedEmbeddingModel

Model = PaddedEmbeddingModel


with open('data.pkl', 'rb') as file:
    instances, schedules, etas = pickle.load(file)

print("generating expert demonstration")
states, actions = [], []
for instance, schedule, eta in zip(instances, schedules, etas):
    eta = iter(eta)
    automaton = Automaton(instance)
    while not automaton.done:
        lane = next(eta)
        state = Model.state_transform(automaton)
        action = Model.inverse_action_transform(automaton, lane)
        automaton.step(lane)

        states.append(state)
        actions.append(action)

train_states = torch.vstack(states)
train_actions = torch.vstack(actions)
train_set = TensorDataset(train_states, train_actions)

in_shape = train_states[0].shape
model = Model(in_shape)
model.cuda()

epochs = 10
batch_size = 10
data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print("\ntraining model\n")
model.train()
for i in range(epochs):
    loss_total = 0
    print(f'epoch: {i}')
    for s, a in tqdm(data_loader, leave=False):
        optimizer.zero_grad()
        logit = model(s)
        loss = F.binary_cross_entropy_with_logits(logit, a)
        loss.backward()
        loss_total += loss
        optimizer.step()
    print(f"loss: {loss_total.item()}\n")


print("evaluating on training data")
model_eval = evaluate(instances, lambda automaton: \
               Model.action_transform(automaton, model(Model.state_transform(automaton))))

obj_total = 0
for schedule in schedules:
    obj_total += schedule['obj']
opt_eval = obj_total / len(schedules)

print(f"mean model obj / mean optimal obj = {model_eval} / {opt_eval} = {model_eval / opt_eval}")
