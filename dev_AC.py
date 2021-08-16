import numpy as np
import torch
import gym
from torch import nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.distributions.normal import Normal
import torch.nn.functional as F


def t(x):
    if type(x) == torch.Tensor:
        return x
    else:
        return torch.from_numpy(x).float()


class Actor(nn.Module):
    def __init__(self, state_dim):
        super().__init__()

        self.fc_1 = nn.Linear(state_dim, 32)
        self.fc_2 = nn.Linear(32, 8)
        self.mu = nn.Linear(8, 2)
        self.sigma = nn.Linear(8, 2)

        self.checkpoint_file ='models/actor_'

    def forward(self, x):
        x = t(x)
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))

        mu = self.mu(x)
        sigma = self.sigma(x)

        sigma = torch.clamp(sigma, min=1e-5, max=1)
        return mu, sigma

    def sample_normal(self, state, reparameterize=False):
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        action = torch.tanh(actions)
        log_probs = probabilities.log_prob(actions)
        log_probs -= torch.log(1 - action.pow(2) + 1e-7)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs

    def save_checkpoint(self,iteration):
        torch.save(self.state_dict(), f'{self.checkpoint_file}{iteration}.pth')

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()

        self.fc_1 = nn.Linear(state_dim, 64)
        self.fc_2 = nn.Linear(64, 32)
        self.fc_3 = nn.Linear(32, 1)

        self.checkpoint_file = 'models/critic_'
    def forward(self, x):
        x = t(x)
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = self.fc_3(x)
        return x


    def save_checkpoint(self,iteration):
        torch.save(self.state_dict(), f'{self.checkpoint_file}{iteration}.pth')

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

class Memory():
    def __init__(self):
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def add(self, log_prob, value, reward, done):
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()

    def _zip(self):
        return zip(self.log_probs,
                   self.values,
                   self.rewards,
                   self.dones)

    def __iter__(self):
        for data in self._zip():
            return data

    def reversed(self):
        for data in list(self._zip())[::-1]:
            yield data

    def __len__(self):
        return len(self.rewards)


def train(memory, q_val):
    values = torch.stack(memory.values)
    q_vals = np.zeros((len(memory), 1))


    for i, (_, _, reward, done) in enumerate(memory.reversed()):
        q_val = reward + gamma * q_val * (1.0 - done)
        q_vals[len(memory) - 1 - i] = q_val

    advantage = torch.Tensor(q_vals) - values

    critic_loss = advantage.pow(2).mean()
    adam_critic.zero_grad()
    critic_loss.backward()
    adam_critic.step()

    actor_loss = (-torch.stack(memory.log_probs) * advantage.detach()).mean()
    adam_actor.zero_grad()
    actor_loss.backward()
    adam_actor.step()

if __name__ == '__main__':
    env = gym.make("LunarLanderContinuous-v2")
    state_dim = env.observation_space.shape[0]
    actor = Actor(state_dim)
    critic = Critic(state_dim)
    adam_actor = torch.optim.Adam(actor.parameters(), lr=1e-3)
    adam_critic = torch.optim.Adam(critic.parameters(), lr=1e-3)
    gamma = 0.99
    memory = Memory()
    max_steps = 200
    episode_rewards = []
    for i in tqdm(range(100_000)):
        done = False
        total_reward = 0
        state = env.reset()
        steps = 0

        while not done:
            action, log_probs = actor.sample_normal(state)
            next_state, reward, done, info = env.step(action.detach().cpu().numpy())
            total_reward += reward
            steps += 1
            memory.add(log_probs, critic(t(state)), reward, done)

            state = next_state

            # train if done or num steps > max_steps
            if done or (steps % max_steps == 0):
                last_q_val = critic(t(next_state)).detach().data.numpy()
                train(memory, last_q_val)
                memory.clear()

        episode_rewards.append(total_reward)

        if i % 100 == 0:
            actor.save_checkpoint(i)
            critic.save_checkpoint(i)
            plt.scatter(np.arange(len(episode_rewards)), episode_rewards, s=2)
            plt.title("Total reward per episode (episodic)")
            plt.ylabel("reward")
            plt.xlabel("episode")
            plt.show()