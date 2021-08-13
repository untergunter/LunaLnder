import gym

import numpy as np
import pandas as pd
import random
import matplotlib
import matplotlib.pyplot as plt
from random import choice
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ipywidgets import IntProgress
from tqdm import tqdm
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm

import sqlite3


def update_db(iteration,total_reward,type_of,actions_space_size):
    statement = f"INSERT INTO results VALUES " +\
                f"({iteration},{total_reward},'{type_of}',{actions_space_size})"
    with sqlite3.connect('model_preformance.db') as con:
        cur = con.cursor()
        cur.execute(statement)
        con.commit()

def get_model_performance_by_type(model_type:str):
    statement = f'select * from results where type="{model_type}" order by iteration asc'
    with sqlite3.connect('model_preformance.db') as con:
        results = pd.read_sql(statement,con)
    return results

def calc_Bellman(rewards,gamma):
    Bellman_rewards = []
    rwords_for_long_train = np.array([])
    for reword in rewards[::-1]:
        rwords_for_long_train = np.append(rwords_for_long_train,np.array([reword]))
        Bellman_rewards.append(np.sum(rwords_for_long_train))
        rwords_for_long_train *= gamma
    Bellman_rewards = Bellman_rewards[::-1]
    return Bellman_rewards

def make_actions():
    actions = [(round(0.1*i,2),round(0.1 * j,2)) for i in range(1, 11) for j in range(-10,11)]
    actions.append((-1, 0))
    return actions


class PolicyAgent:

    def __init__(self):
        self.state_action_score = defaultdict(lambda: defaultdict(list))  # state : {act:[scores]...act:[scores]}
        self.actions = make_actions()

    def normlize(self, state):
        return tuple(np.round(state, decimals=2))

    def exploit(self, state):
        state = self.normlize(state)
        best_score = - np.inf
        best_action = None
        for action in self.state_action_score[state]:
            if len(self.state_action_score[state][action]) >0:
                score = np.mean(self.state_action_score[state][action])
                if score < best_score:
                    best_action = score
                    best_action = action
        if best_action is None:  # never been here before - i dont know what to do !
            action = self.explore(state)
        return action

    def explore(self, state):
        state = self.normlize(state)
        minimal_number_of_observations = np.inf
        potential_steps = []
        for action in self.actions:
            number_of_observations = len(self.state_action_score[state][action])

            if number_of_observations < minimal_number_of_observations:
                potential_steps = [action]
                minimal_number_of_observations = number_of_observations

            elif number_of_observations == minimal_number_of_observations:
                potential_steps.append(action)

        return choice(potential_steps)

    def update(self, state, action, score):
        state = self.normlize(state)
        score = float(score)
        self.state_action_score[state][action].append(score)

    def update_multiple(self, states, actions, scores):
        for state, action, score in zip(states, actions, scores):
            self.update(state, action, score)


def train(steps: int = 100_000):
    env = gym.make('LunarLanderContinuous-v2')
    env.seed(0)
    iterations, total_scores = [], []
    agent = PolicyAgent()
    for step in range(steps):

        states, actions, rewards = [], [], []
        state = env.reset()
        over = False
        rounds = 0

        # play to learn
        while (not over) & (rounds < 201):
            action = agent.explore(state)
            next_state, reward, over, _ = env.step(action)
            actions.append(action)
            states.append(state)
            rewards.append(reward)
            rounds += 1
            state = next_state
            agent.update(state,action,reward)
        # learn from the game
        rewards = calc_Bellman(rewards, 0.9)
        agent.update_multiple(states, actions, rewards)

        # play to win
        state = env.reset()
        over = False
        rounds = 0
        total_score = 0

        # play to learn
        while (not over) & (rounds < 201):
            action = agent.exploit(state)
            next_state, reward, over, _ = env.step(action)
            total_score += reward
            rounds += 1
            state = next_state
        iterations.append(step)
        total_scores.append(total_score)
        print(f'iteration:{step},score:{round(total_score, 2)}')
        update_db(step, total_score, 'policy', len(agent.actions))
    return agent

if __name__=='__main__':
    lender = train()