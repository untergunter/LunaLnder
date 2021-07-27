import gym

import numpy as np
import sys
import random

import pandas as pd
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

def calc_Bellman(rewards,gamma):
    Bellman_rewards = []
    rwords_for_long_train = np.array([])
    for reword in rewards[::-1]:
        rwords_for_long_train = np.append(rwords_for_long_train,np.array([reword]))
        Bellman_rewards.append(np.sum(rwords_for_long_train))
        rwords_for_long_train *= gamma
    Bellman_rewards = Bellman_rewards[::-1]
    return Bellman_rewards

class QNet(nn.Module):
    def __init__(self, action_space,device):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(10, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 1)
        self.action_space = torch.from_numpy(np.array(action_space))
        self.device = device

    def forward(self, x):
        """ input = concat(state,action)"""
        x = torch.from_numpy(x,dtype='float32').to(self.device) if type(x)==np.ndarray else x.float().to(self.device)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def concat_actions_states(self, state):
        repeated = torch.tensor(state).repeat(self.action_space.shape[0], 1)
        together = torch.tensor(np.concatenate([repeated, self.action_space], axis=1))
        return together

    def get_next_move(self, state):
        state_and_actions = self.concat_actions_states(state)
        with torch.no_grad():
            predicted_score = self(state_and_actions)
        best_index = torch.argmax(predicted_score).item()
        wanted_action = tuple( i for i in self.action_space[best_index])
        return wanted_action

def train(steps_to_train:int,gamma:float=0.9):
    actions = [(i, j) for i in range(0, 2) for j in range(-1, 2)]
    actions.append((-1, 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QNet(actions,device).to(device)
    env = gym.make('LunarLanderContinuous-v2')
    env.seed(0)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.L1Loss()

    for _ in tqdm(range(steps_to_train)):

        states,rewards = [],[]
        state = env.reset()
        over = False
        rounds = 0

        while (not over) & (rounds<201) :
            action = model.get_next_move(state)
            next_state, reward, over, _ = env.step(action)
            states.append(np.concatenate([state,action]))
            rewards.append(reward)
            rounds += 1
            state = next_state


        x = torch.tensor(states)
        score_pred = model(x)
        score_pred = score_pred.view(score_pred.shape[0])
        optimizer.zero_grad()
        y = torch.tensor(calc_Bellman(rewards,0.9)).to(device)
        loss = loss_fn(score_pred, y)
        loss.backward()
        optimizer.step()

    return model

if __name__ == '__main__':
    for gamma in (0.1 * i for i in range(1,14)):
        trined_model = train(1_000,gamma)
        actions = [(i, j) for i in range(0, 2) for j in range(-1, 2)]
        actions.append((-1, 0))
        with torch.no_grad():
            env = gym.make('LunarLanderContinuous-v2')
            env.seed(1)
            scores = []
            for time in range(100):
                state = env.reset()
                over = False
                rounds = 0
                total_score = 0

                while (not over) & (rounds<201) :
                    action = trined_model.get_next_move(state)
                    next_state, reward, over, _ = env.step(action)
                    total_score += reward
                    rounds += 1
                    state = next_state
                scores.append(total_score)
            results = pd.DataFrame({'iteration':[i for i in range(1,len(scores)+1)],
                          'total_reward':scores,
                          'type':[f'100_deep_q_bellman_{gamma}' for i in range(len(scores))],
                         'action_space_':[len(actions) for i in range(len(scores))]})
            results.to_parquet(f'deep_q_100_100_b_g_{gamma}.parquet.gzip',compression='gzip',index=False)