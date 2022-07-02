"""
All decisions are made in here.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Updated by Ashitemaru, 2022/07/02.
"""

import numpy as np
import pandas as pd

class RL(object):
    def __init__(
        self,
        action_space,
        learning_rate = 0.01,
        reward_decay = 0.9,
        e_greedy = 0.9
    ):
        self.actions = action_space
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy

        self.q_table = pd.DataFrame(columns = self.actions, dtype = np.float64)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # Append new state to Q table
            self.q_table.loc[state] = [0] * len(self.actions)

    def choose_action(self, state):
        self.check_state_exist(state)

        # Action selection
        if np.random.rand() < self.epsilon: # Choose best action
            state_action = self.q_table.loc[state, :]
            # Some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else: # Choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, *args):
        pass

# On policy - Sarsa
class SarsaTable(RL):
    def __init__(
        self,
        actions,
        learning_rate = 0.01,
        reward_decay = 0.9,
        e_greedy = 0.9
    ):
        super(SarsaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self, state, action, reward, next_state, next_action):
        self.check_state_exist(next_state)

        q_predict = self.q_table.loc[state, action]
        if next_state != "terminal":
            q_target = reward + self.gamma * self.q_table.loc[next_state, next_action]
        else:
            q_target = reward
            
        self.q_table.loc[state, action] += self.lr * (q_target - q_predict)

# Optimized on policy - Sarsa lambda
class SarsaLambdaTable(RL):
    def __init__(
        self,
        actions,
        learning_rate = 0.01,
        reward_decay = 0.9,
        e_greedy = 0.9,
        trace_decay = 0.9
    ):
        super(SarsaLambdaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)
        self.lambda_ = trace_decay
        self.e_table = self.q_table.copy()

    def learn(self, state, action, reward, next_state, next_action):
        self.check_state_exist(next_state)

        q_predict = self.q_table.loc[state, action]
        if next_state != "terminal":
            q_target = reward + self.gamma * self.q_table.loc[next_state, next_action]
        else:
            q_target = reward
            
        error = q_target - q_predict

        self.e_table.loc[state, action] += 1

        self.q_table += self.lr * error * self.e_table
        self.e_table *= self.gamma * self.lambda_ # Decay

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # Append new state to Q table & E table
            self.q_table.loc[state] = [0] * len(self.actions)
            self.e_table.loc[state] = [0] * len(self.actions)
