"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Updated by Ashitemaru, 2022/07/02.
"""

import numpy as np
import pandas as pd

class QLearningTable:
    def __init__(
        self,
        actions,
        learning_rate = 0.01,
        reward_decay = 0.9,
        e_greedy = 0.9
    ):
        self.actions = actions # Action list
        self.lr = learning_rate # Alpha
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns = self.actions, dtype = np.float64) # Init Q table

    def choose_action(self, state):
        self.check_state_exist(state)

        # Action selection
        if np.random.uniform() < self.epsilon: # Choose best action
            state_action = self.q_table.loc[state, :]
            # Some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else: # Choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, state, action, reward, next_state):
        self.check_state_exist(next_state)
        q_predict = self.q_table.loc[state, action]

        if next_state != "terminal":
            q_target = reward + self.gamma * self.q_table.loc[next_state, :].max()
        else:
            q_target = reward

        self.q_table.loc[state, action] += self.lr * (q_target - q_predict) # Update

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # Append new state to Q table
            self.q_table.loc[state] = [0] * len(self.actions)