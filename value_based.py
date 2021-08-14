import gym
import numpy as np
import random
import os
from dev import update_db,make_actions
import torch
import torch.nn as nn
import torch.nn.functional as F

def calc_Bellman(rewards,gamma):
    Bellman_rewards = []
    rewords_for_long_train = np.array([])
    for reword in rewards[::-1]:
        rewords_for_long_train = np.append(rewords_for_long_train,np.array([reword]))
        Bellman_rewards.append(np.sum(rewords_for_long_train))
        rewords_for_long_train *= gamma
    Bellman_rewards = Bellman_rewards[::-1]
    return Bellman_rewards

class QNet(nn.Module):
    def __init__(self, device,search_rate:int=1):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 1)
        self.action_space = torch.from_numpy(np.array(make_actions()))
        self.action_space_size = len(make_actions())
        self.device = device
        self.search_rate = search_rate

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

    def get_next_move(self, state, search=True):
        take_random = random.random() > self.search_rate
        if take_random and search:
            wanted_action = tuple( i for i in random.choice(self.action_space))
        else:
            state_and_actions = self.concat_actions_states(state)
            with torch.no_grad():
                predicted_score = self(state_and_actions)
            best_index = torch.argmax(predicted_score).item()
            wanted_action = tuple( i for i in self.action_space[best_index])
        return wanted_action

def train(steps_to_train:int,model_name:str,gamma:float=0.9):
    actions = [(i, j) for i in range(0, 2) for j in range(-1, 2)]
    actions.append((-1, 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QNet(device).to(device)
    env = gym.make('LunarLanderContinuous-v2')
    env.seed(0)
    test_env = gym.make('LunarLanderContinuous-v2')
    test_env.seed(1)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.L1Loss()

    for step in range(steps_to_train):
        model.train()
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
        y_1 = torch.tensor(rewards).to(device)
        loss_1 = loss_fn(score_pred, y_1)
        loss_1.backward(retain_graph=True)
        optimizer.zero_grad()
        y_2 = torch.tensor(calc_Bellman(rewards,gamma)).to(device)
        loss_2 = loss_fn(score_pred, y_2)
        loss_2.backward()
        optimizer.step()
        total_loss = loss_1.item()+loss_2.item()
        print(f'loss={total_loss}')

        if step%100 == 0:

            if model.search_rate > 0.1:
                model.search_rate -= 0.005
            model.eval()
            batch_score = 0
            with torch.no_grad():
                for time in range(10):
                    state = test_env.reset()
                    over = False
                    rounds = 0
                    total_score = 0
                    actions,scores = [],[]
                    while (not over) & (rounds < 201):
                        action = model.get_next_move(state,search=False)
                        actions.append(action)
                        next_state, reward, over, _ = env.step(action)
                        scores.append(reward)
                        total_score += reward
                        rounds += 1
                        state = next_state
                    batch_score += total_score
            mean_batch_score = batch_score / 10
            model_path = f'models{os.sep}_{model_name}_{step}.pth'
            torch.save(model.state_dict(),model_path)
            update_db(step, mean_batch_score, model_name, model.action_space_size)
    return model

if __name__ == '__main__':
    training = train(60_000,"value_search_small_net")

