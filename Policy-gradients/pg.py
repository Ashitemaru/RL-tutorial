"""
This part of code is the reinforcement learning brain, which is a brain of the agent.
All decisions are made in here.
Policy Gradient, Reinforcement Learning.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Updated by Ashitemaru, 2022/07/18.
"""

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

np.random.seed(1)
tf.set_random_seed(1)

class PolicyGradient:
    def __init__(
        self,
        n_actions,
        n_features,
        learning_rate = 0.01,
        reward_decay = 0.95,
        output_graph = False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        self.epoch_state_list = []
        self.epoch_action_list = []
        self.epoch_reward_list = []

        self._build_net()

        self.session = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.session.graph)

        self.session.run(tf.global_variables_initializer())

    def _build_net(self):
        with tf.name_scope("inputs"):
            self.tf_state_list = tf.placeholder(tf.float32, [None, self.n_features], name = "state_list")
            self.tf_action_list = tf.placeholder(tf.int32, [None, ], name = "action_list")
            self.tf_reward_list = tf.placeholder(tf.float32, [None, ], name = "reward_list")

        # FC1
        layer = tf.keras.layers.Dense(
            units = 10,
            activation = tf.nn.tanh,
            kernel_initializer = tf.random_normal_initializer(mean = 0, stddev = 0.3),
            bias_initializer = tf.constant_initializer(0.1),
            name = "fc1"
        )(self.tf_state_list)

        # FC2
        all_action = tf.keras.layers.Dense(
            units = self.n_actions,
            activation = None,
            kernel_initializer = tf.random_normal_initializer(mean = 0, stddev = 0.3),
            bias_initializer = tf.constant_initializer(0.1),
            name = "fc2"
        )(layer)

        self.all_action_prob = tf.nn.softmax(all_action, name = "action_prob")

        with tf.name_scope("loss"):
            # To maximize total reward (log_p * R) is to minimize -(log_p * R)
            # And the tf only have minimize(loss)
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits = all_action,
                labels = self.tf_action_list) # This is negative log of chosen action

            loss = tf.reduce_mean(neg_log_prob * self.tf_reward_list) # Reward guided loss

        with tf.name_scope("train"):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def choose_action(self, state):
        prob_weights = self.session.run(
            self.all_action_prob,
            feed_dict = { self.tf_state_list: state[np.newaxis, :] })
        action = np.random.choice(range(prob_weights.shape[1]), p = prob_weights.ravel())
        return action

    def store_transition(self, state, action, reward):
        self.epoch_state_list.append(state)
        self.epoch_action_list.append(action)
        self.epoch_reward_list.append(reward)

    def learn(self):
        # Discount and normalize episode reward
        discounted_epoch_reward_list = self._discount_and_norm_rewards()

        # Train on episode
        self.session.run(self.train_op, feed_dict = {
            self.tf_state_list: np.vstack(self.epoch_state_list), # shape = [None, n_obs]
            self.tf_action_list: np.array(self.epoch_action_list), # shape = [None, ]
            self.tf_reward_list: discounted_epoch_reward_list, # shape = [None, ]
        })

        self.epoch_state_list = []
        self.epoch_action_list = []
        self.epoch_reward_list = []

        return discounted_epoch_reward_list

    def _discount_and_norm_rewards(self):
        # Discount episode rewards
        discounted_epoch_reward_list = np.zeros_like(self.epoch_reward_list)

        # Accumulate the rewards by the parameter "gamma"
        running_add = 0
        for t in reversed(range(0, len(self.epoch_reward_list))):
            running_add = running_add * self.gamma + self.epoch_reward_list[t]
            discounted_epoch_reward_list[t] = running_add

        # Normalize episode rewards
        discounted_epoch_reward_list -= np.mean(discounted_epoch_reward_list)
        discounted_epoch_reward_list /= np.std(discounted_epoch_reward_list)
        return discounted_epoch_reward_list