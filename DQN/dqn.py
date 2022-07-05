"""
This part of code is the Deep Q Network (DQN) brain.
View the tensorboard picture about this DQN structure on: https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/4-3-DQN3/#modification
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
Using:
Tensorflow: r1.2

Updated by Ashitemaru, 2022/07/05.
"""

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

np.random.seed(1)
tf.set_random_seed(1)

# Deep Q Network
class DeepQNetwork:
    def __init__(
        self,
        n_actions,
        n_features,
        learning_rate = 0.01,
        reward_decay = 0.9,
        e_greedy = 0.9,
        replace_target_iter = 300,
        memory_size = 500,
        batch_size = 32,
        e_greedy_increment = None,
        output_graph = False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # Total learning step
        self.learn_step_counter = 0

        # Initialize zero memory [state, action, reward, next_state]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # Consist of [target_net, evaluate_net]
        self._build_net()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "target_net")
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "eval_net")

        with tf.variable_scope("hard_replacement"):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.session = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            tf.summary.FileWriter("logs/", self.session.graph)

        self.session.run(tf.global_variables_initializer())
        self.cost_history = []

    def _build_net(self):
        # Inputs
        self.state = tf.placeholder(tf.float32, [None, self.n_features], name = "state")
        self.next_state = tf.placeholder(tf.float32, [None, self.n_features], name = "next_state")
        self.reward = tf.placeholder(tf.float32, [None, ], name = "reward")
        self.action = tf.placeholder(tf.int32, [None, ], name = "action")

        w_initializer = tf.random_normal_initializer(0., 0.3)
        b_initializer = tf.constant_initializer(0.1)

        # Evaluate network
        with tf.variable_scope("eval_net"):
            eval_layer1 = tf.layers.dense(
                inputs = self.state,
                units = 20,
                activation = tf.nn.relu,
                kernel_initializer = w_initializer,
                bias_initializer = b_initializer,
                name = "eval_layer1"
            )
            self.q_eval = tf.layers.dense(
                inputs = eval_layer1,
                units = self.n_actions,
                kernel_initializer = w_initializer,
                bias_initializer = b_initializer,
                name = "q_eval"
            )

        # Target network
        with tf.variable_scope("target_net"):
            target_layer1 = tf.layers.dense(
                inputs = self.next_state,
                units = 20,
                activation = tf.nn.relu,
                kernel_initializer = w_initializer,
                bias_initializer = b_initializer,
                name = "target_layer1"
            )
            self.q_next = tf.layers.dense(
                inputs = target_layer1,
                units = self.n_actions,
                kernel_initializer = w_initializer,
                bias_initializer = b_initializer,
                name = "q_next"
            )

        with tf.variable_scope("q_target"):
            q_target = self.reward + self.gamma * tf.reduce_max(self.q_next, axis = 1, name = "q_max_next_state")
            self.q_target = tf.stop_gradient(q_target)

        with tf.variable_scope("q_eval"):
            action_indices = tf.stack([tf.range(tf.shape(self.action)[0], dtype = tf.int32), self.action], axis = 1)
            self.q_eval_wrt_action = tf.gather_nd(params = self.q_eval, indices = action_indices)

        with tf.variable_scope("loss"):
            self.loss = tf.reduce_mean(tf.squared_difference(
                self.q_target, self.q_eval_wrt_action,
                name = "error"
            ))
        
        with tf.variable_scope("train"):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    def store_transition(self, state, action, reward, next_state):
        if not hasattr(self, "memory_counter"):
            self.memory_counter = 0

        transition = np.hstack((state, [action, reward], next_state))

        # Replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, state):
        # To have batch dimension when feed into TF placeholder
        state = state[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # Forward feed the state and get Q value for every actions
            actions_value = self.session.run(self.q_eval, feed_dict = { self.state: state })
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # Check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.session.run(self.target_replace_op)
            print("Target params replaced")

        # Sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size = self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size = self.batch_size)
        batch_memory = self.memory[sample_index, :]

        _, cost = self.session.run(
            [self._train_op, self.loss],
            feed_dict = {
                self.state: batch_memory[:, : self.n_features],
                self.action: batch_memory[:, self.n_features],
                self.reward: batch_memory[:, self.n_features + 1],
                self.next_state: batch_memory[:, -self.n_features: ],
            })

        self.cost_history.append(cost)

        # Increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment \
            if self.epsilon < self.epsilon_max \
            else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_history)), self.cost_history)
        plt.ylabel("Cost")
        plt.xlabel("Training steps")
        plt.show()