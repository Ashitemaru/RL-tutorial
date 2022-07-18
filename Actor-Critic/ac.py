"""
Actor-Critic using TD-error as the Advantage, Reinforcement Learning.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/

@warning: May not converge

Updated by Ashitemaru, 2022/07/18.
"""

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

np.random.seed(2)
tf.set_random_seed(2)

class Actor(object):
    def __init__(self, session, n_features, n_actions, lr = 0.001):
        self.session = session

        self.state = tf.placeholder(tf.float32, [1, n_features], "actor_state")
        self.action = tf.placeholder(tf.int32, None, "actor_action")
        self.td_error = tf.placeholder(tf.float32, None, "actor_td_error")

        with tf.variable_scope("actor"):
            actor_l1 = tf.keras.layers.Dense(
                units = 20,
                activation = tf.nn.relu,
                kernel_initializer = tf.random_normal_initializer(0., .1),
                bias_initializer = tf.constant_initializer(.1),
                name = "actor_l1"
            )(self.state)

            self.action_prob = tf.keras.layers.Dense(
                units = n_actions,
                activation = tf.nn.softmax, # Transform it into probability
                kernel_initializer = tf.random_normal_initializer(0., .1),
                bias_initializer = tf.constant_initializer(0.1),
                name = "actor_action_prob"
            )(actor_l1)

        with tf.variable_scope("action_value"):
            log_prob = tf.log(self.action_prob[0, self.action])
            # TD Error - scalar
            self.action_value = tf.reduce_mean(log_prob * self.td_error) # Advantage (TD_error) guided loss

        with tf.variable_scope("actor_train"):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.action_value)

    def learn(self, state, action, td_error):
        state = state[np.newaxis, :]
        _, action_value = self.session.run([self.train_op, self.action_value], feed_dict = {
            self.state: state,
            self.action: action,
            self.td_error: td_error
        })
        return action_value

    def choose_action(self, state):
        state = state[np.newaxis, :]
        probs = self.session.run(self.action_prob, feed_dict = { self.state: state })
        return np.random.choice(np.arange(probs.shape[1]), p = probs.ravel())

class Critic(object):
    def __init__(self, session, n_features, lr = 0.01, gamma = 0.9):
        self.session = session
        self.gamma = gamma

        self.state = tf.placeholder(tf.float32, [1, n_features], "critic_state")
        self.next_state_value = tf.placeholder(tf.float32, [1, 1], "critic_next_state_value")
        self.reward = tf.placeholder(tf.float32, None, "critic_reward")

        with tf.variable_scope("critic"):
            critic_l1 = tf.keras.layers.Dense(
                units = 20,
                activation = tf.nn.relu,
                # Have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer = tf.random_normal_initializer(0., .1),
                bias_initializer = tf.constant_initializer(.1),
                name = "critic_l1"
            )(self.state)

            self.state_value = tf.keras.layers.Dense(
                units = 1,
                activation = None,
                kernel_initializer = tf.random_normal_initializer(0., .1),
                bias_initializer = tf.constant_initializer(.1),
                name = "critic_state_value"
            )(critic_l1)

        with tf.variable_scope("critic_squared_td_error"):
            self.td_error = self.reward + self.gamma * self.next_state_value - self.state_value
            self.loss = tf.square(self.td_error)

        with tf.variable_scope("critic_train"):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, state, reward, next_state):
        state, next_state = state[np.newaxis, :], next_state[np.newaxis, :]

        next_state_value = self.session.run(self.state_value, feed_dict = { self.state: next_state })
        td_error, _ = self.session.run([self.td_error, self.train_op], feed_dict = {
            self.state: state,
            self.next_state_value: next_state_value,
            self.reward: reward
        })

        return td_error