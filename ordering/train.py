import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from automaton import Automaton, evaluate
from model import PaddedEmbeddingModel
from util import equalp, plot_schedule


Model = PaddedEmbeddingModel

# data loading
with open('data.pkl', 'rb') as file:
    data = pickle.load(file)
    data = pd.DataFrame.from_records(zip(*data), columns=["instances", "schedules", "etas"])

# create train/test split
data_train, data_test = train_test_split(data, test_size=0.2)

print("generating expert demonstration")
states, actions = [], []
for _, (instance, schedule, eta) in data_train.iterrows():
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
trained_heuristic = lambda automaton: \
               Model.action_transform(automaton, model(Model.state_transform(automaton)))

opt_eval, hat_eval, equal_total = 0, 0, 0
for _, (instance, y_opt, _) in data_test.iterrows():
    y_hat = evaluate(instance, trained_heuristic)
    hat_eval += y_hat['obj']
    opt_eval += y_opt['obj']
    equal_total += int(equalp(y_hat, y_opt))
opt_eval = opt_eval / len(data_test)
hat_eval = hat_eval / len(data_test)

print(f"mean model obj / mean optimal obj = {hat_eval} / {opt_eval} = {hat_eval / opt_eval}")
print(f"optimal / total = {equal_total} / {len(data_test)} = {equal_total / len(data_test)}")
