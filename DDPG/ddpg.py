"""
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.
Pendulum example.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Updated by Ashitemaru, 2022/07/23.
"""

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

np.random.seed(1)
tf.set_random_seed(1)

class DDPG(object):
    def __init__(
        self,
        n_action,
        n_feature,
        action_bound,
        memory_capacity = 10000,
        soft_replace_param = 0.01,
        reward_decay = 0.9,
        actor_lr = 0.001,
        critic_lr = 0.003,
        batch_size = 32,
        output_graph = True,
    ):
        # Memory
        self.memory_capacity = memory_capacity
        self.memory = np.zeros((memory_capacity, n_feature * 2 + n_action + 1), dtype = np.float32)
        self.memory_ptr = 0

        self.session = tf.Session()

        # Params
        self.n_action = n_action
        self.n_feature = n_feature
        self.action_bound = action_bound
        self.tau = soft_replace_param
        self.gamma = reward_decay
        self.batch_size = batch_size

        # Placeholders
        self.state = tf.placeholder(tf.float32, [None, n_feature], "state")
        self.next_state = tf.placeholder(tf.float32, [None, n_feature], "next_state")
        self.reward = tf.placeholder(tf.float32, [None, 1], "reward")

        with tf.variable_scope("actor"):
            self.action = self._build_actor(self.state, scope = "eval", trainable = True)
            next_action = self._build_actor(self.next_state, scope = "target", trainable = False)

        with tf.variable_scope("critic"):
            # Assign self.action = action in memory when calculating Q value for td_error,
            # Otherwise the self.action is from Actor when updating Actor
            q_value = self._build_critic(self.state, self.action, scope = "eval", trainable = True)
            next_q_value = self._build_critic(self.next_state, next_action, scope = "target", trainable = False)

        # Networks parameters
        self.actor_eval_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "actor/eval")
        self.actor_target_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "actor/target")
        self.critic_eval_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "critic/eval")
        self.critic_target_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "critic/target")

        # Target net replacement
        self.soft_replace = [
            tf.assign(t, (1 - self.tau) * t + self.tau * e)
                for t, e in zip(
                    self.actor_target_params + self.critic_target_params,
                    self.actor_eval_params + self.critic_eval_params
                )
        ]

        q_target = self.reward + self.gamma * next_q_value
        td_error = tf.losses.mean_squared_error(labels = q_target, predictions = q_value)
        self.critic_train = tf.train.AdamOptimizer(critic_lr).minimize(td_error, var_list = self.critic_eval_params)

        actor_loss = -tf.reduce_mean(q_value)
        self.actor_train = tf.train.AdamOptimizer(actor_lr).minimize(actor_loss, var_list = self.actor_eval_params)

        self.session.run(tf.global_variables_initializer())

        if output_graph:
            tf.summary.FileWriter("logs/", self.session.graph)

    def choose_action(self, state):
        return self.session.run(self.action, feed_dict = { self.state: state[np.newaxis, :] })[0]

    def learn(self):
        # Soft target replacement
        self.session.run(self.soft_replace)

        indices = np.random.choice(self.memory_capacity, size = self.batch_size)
        batch = self.memory[indices, :]
        batch_state = batch[:, : self.n_feature]
        batch_action = batch[:, self.n_feature: self.n_feature + self.n_action]
        batch_reward = batch[:, -self.n_feature - 1: -self.n_feature]
        batch_next_state = batch[:, -self.n_feature: ]

        self.session.run(self.actor_train, feed_dict = {
            self.state: batch_state
        })
        self.session.run(self.critic_train, feed_dict = {
            self.state: batch_state,
            self.action: batch_action,
            self.reward: batch_reward,
            self.next_state: batch_next_state
        })

    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, action, [reward], next_state))
        index = self.memory_ptr % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_ptr += 1

    def _build_actor(self, state_input_placeholder, scope, trainable):
        with tf.variable_scope(scope):
            actor_l1 = tf.keras.layers.Dense(
                units = 30,
                activation = tf.nn.relu,
                name = "actor_l1",
                trainable = trainable
            )(state_input_placeholder)
            action = tf.keras.layers.Dense(
                units = self.n_action,
                activation = tf.nn.tanh,
                name = "action",
                trainable = trainable
            )(actor_l1)

            return tf.multiply(action, self.action_bound, name = "scaled_action")

    def _build_critic(self, state_input_placeholder, action_input_placeholder, scope, trainable):
        with tf.variable_scope(scope):
            units = 30
            state_weight_matrix = tf.get_variable(
                "state_weight_matrix", [self.n_feature, units],
                trainable = trainable
            )
            action_weight_matrix = tf.get_variable(
                "action_weight_matrix", [self.n_action, units],
                trainable = trainable
            )
            bias = tf.get_variable(
                "bias", [1, units],
                trainable = trainable
            )
            critic_l1 = tf.nn.relu(
                tf.matmul(state_input_placeholder, state_weight_matrix) +
                tf.matmul(action_input_placeholder, action_weight_matrix) +
                bias
            )

            return tf.keras.layers.Dense(units = 1, trainable = trainable)(critic_l1)