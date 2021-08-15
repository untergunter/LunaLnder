import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import gym
from scipy.optimize import minimize
import random



class Critic(nn.Module):
    def __init__(self, device):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 1)
        self.device = device


    def forward(self, x):
        """ input = concat(state,action) """
        x = torch.from_numpy(x).float().to(self.device) if type(x)==np.ndarray else x.float().to(self.device)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x).float()
        return x


def calc_Bellman(rewards,gamma):
    Bellman_rewards = []
    rewords_for_long_train = np.array([])
    for reword in rewards[::-1]:
        rewords_for_long_train = np.append(rewords_for_long_train,np.array([reword]))
        Bellman_rewards.append(np.sum(rewords_for_long_train))
        rewords_for_long_train *= gamma
    Bellman_rewards = Bellman_rewards[::-1]
    return Bellman_rewards


class Agent():

    def __init__(self,model,search_rate:int=1):
        self.bounds = [(-1,1),(-1,1)]
        self.state = None
        self.search_rate = search_rate
        self.gamma = 0.9
        self.loss_fn = nn.L1Loss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.state = None

    def set_state(self,new_state):
        if type(new_state) == np.ndarray:
            self.state = new_state

    def get_optimal_action(self):
        action = np.random.uniform(-1,1,2)
        optimal_action = minimize(self.predict_negative,
                                  action, bounds=self.bounds,
                                  method='SLSQP').x
        return optimal_action

    def decrease_search_rate(self):
        if self.search_rate > 0.1:
            self.search_rate -= 0.05

    def get_next_step(self,state):
        self.set_state(state)
        take_random = random.random() < self.search_rate
        if take_random:
            action = self.get_random_action()
        else:
            action = self.get_optimal_action()
        return action

    def get_random_action(self):
        return np.random.uniform(-1,1,2)

    def predict_negative(self,x):
        state_and_action = np.concatenate([self.state,x])
        score = - self.model(state_and_action).cpu().detach().numpy()
        return score

    def train_model_on_single_game(self,action_states,rewards):
        x = torch.tensor(action_states)
        score_pred = self.model(x)
        score_pred = score_pred.view(score_pred.shape[0])
        self.optimizer.zero_grad()
        y_1 = torch.tensor(rewards).float().to(self.device)
        loss_1 = self.loss_fn(score_pred, y_1)
        loss_1.backward(retain_graph=True)
        # self.optimizer.zero_grad()
        # y_2 = torch.tensor(calc_Bellman(rewards,self.gamma)).float().to(self.device)
        # y_2 = torch.reshape(y_2, ((y_2.size())[0], 1))
        # loss_2 = self.loss_fn(score_pred, y_2)
        # loss_2.backward()
        # self.optimizer.step()
        # total_loss = loss_1.item()+loss_2.item()
        # return total_loss
        return loss_1.item()

def main(steps_to_train:int=10_000):
    env = gym.make('LunarLanderContinuous-v2')
    model = Critic(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    player = Agent(model)
    for step in range(steps_to_train):
        states, rewards = [], []
        state = env.reset()
        over = False
        rounds = 0

        while (not over) & (rounds < 201):
            action = player.get_next_step(state)
            next_state, reward, over, _ = env.step(action)
            states.append(np.concatenate([state, action]))
            rewards.append(reward)
            rounds += 1
            state = next_state

        total_loss = player.train_model_on_single_game(states,rewards)
        train_score = np.sum(rewards)
        train_length = len(rewards)

        states, rewards = [], []
        state = env.reset()
        over = False
        rounds = 0

        while (not over) & (rounds < 201):
            player.set_state(state)
            action = player.get_optimal_action()
            next_state, reward, over, _ = env.step(action)
            states.append(np.concatenate([state, action]))
            rewards.append(reward)
            rounds += 1
            state = next_state

        # total_test_loss = player.train_model_on_single_game(states, rewards)
        test_score = np.sum(rewards)
        test_length = len(rewards)
        print(f'{step} train: loss={int(total_loss)} score={int(train_score)} {train_length} steps,test: score={int(test_score)} {test_length} steps')


if __name__ == '__main__':
    main()