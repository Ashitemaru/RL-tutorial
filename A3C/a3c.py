"""
Asynchronous Advantage Actor Critic (A3C) with continuous action space, Reinforcement Learning.
The Pendulum example.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Updated by Ashitemaru, 2022/07/23.
"""

import numpy as np
import tensorflow.compat.v1 as tf
import gym

# TODO: Ugly globals
g_epoch = 0
g_running_reward = []

tf.disable_v2_behavior()

np.random.seed(2)
tf.set_random_seed(2)

class ActorCriticNet(object):
    GLOBAL_NET_SCOPE = "global_net"

    def __init__(
        self,
        n_feature,
        n_action,
        action_bound,
        scope,
        session,
        actor_optimizer,
        critic_optimizer, # TODO: Optimizers as parameters?
        global_net = None,
        entropy_beta = 0.01,
    ):
        self.session = session
        self.n_action = n_action

        if scope == ActorCriticNet.GLOBAL_NET_SCOPE: # Global net
            with tf.variable_scope(scope):
                self.state = tf.placeholder(tf.float32, [None, n_feature], "state")
                self.actor_params, self.critic_params = self._build_net(scope)[-2: ]
        else: # Local net
            with tf.variable_scope(scope):
                self.state = tf.placeholder(tf.float32, [None, n_feature], "state")
                self.action_history = tf.placeholder(tf.float32, [None, n_action], "action")
                self.q_value_target = tf.placeholder(tf.float32, [None, 1], "q_target")

                mu, sigma, self.q_value, self.actor_params, self.critic_params = self._build_net(scope)

                td_error = tf.subtract(self.q_value_target, self.q_value, name = "td_error")
                with tf.name_scope("critic_loss"):
                    self.critic_loss = tf.reduce_mean(tf.square(td_error))

                with tf.name_scope("wrapped_actor_output"):
                    mu *= action_bound[1]
                    sigma += 1e-4

                normal_dist = tf.distributions.Normal(mu, sigma)

                with tf.name_scope("actor_loss"):
                    log_prob = normal_dist.log_prob(self.action_history)
                    exp_v = log_prob * tf.stop_gradient(td_error)
                    entropy = normal_dist.entropy()  # encourage exploration
                    self.exp_v = entropy_beta * entropy + exp_v
                    self.actor_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope("choose_action"):
                    self.action = tf.clip_by_value(
                        tf.squeeze(normal_dist.sample(1), axis = [0, 1]),
                        action_bound[0],
                        action_bound[1],
                    )

                with tf.name_scope("local_grad"):
                    self.actor_grads = tf.gradients(self.actor_loss, self.actor_params)
                    self.critic_grads = tf.gradients(self.critic_loss, self.critic_params)

            with tf.name_scope("sync"):
                with tf.name_scope("pull"):
                    self.pull_actor_params = [
                        local_param.assign(global_param)
                            for local_param, global_param in zip(self.actor_params, global_net.actor_params)
                    ]
                    self.pull_critic_params = [
                        local_param.assign(global_param)
                            for local_param, global_param in zip(self.critic_params, global_net.critic_params)
                    ]

                with tf.name_scope("push"):
                    self.update_actor = actor_optimizer.apply_gradients(
                        zip(self.actor_grads, global_net.actor_params))
                    self.update_critic = critic_optimizer.apply_gradients(
                        zip(self.critic_grads, global_net.critic_params))

    def _build_net(self, scope):
        weight_initializer = tf.random_normal_initializer(0., .1)
        with tf.variable_scope("actor"):
            actor_l1 = tf.keras.layers.Dense(
                units = 200,
                activation = tf.nn.relu6,
                kernel_initializer = weight_initializer,
                name = "actor_l1"
            )(self.state)
            mu = tf.keras.layers.Dense(
                units = self.n_action,
                activation = tf.nn.tanh,
                kernel_initializer = weight_initializer,
                name = "mu"
            )(actor_l1)
            sigma = tf.keras.layers.Dense(
                units = self.n_action,
                activation = tf.nn.softplus,
                kernel_initializer = weight_initializer,
                name = "sigma"
            )(actor_l1)

        with tf.variable_scope("critic"):
            critic_l1 = tf.keras.layers.Dense(
                units = 100,
                activation = tf.nn.relu6,
                kernel_initializer = weight_initializer,
                name = "critic_l1"
            )(self.state)
            q_value = tf.keras.layers.Dense(
                units = 1,
                kernel_initializer = weight_initializer,
                name = "q_value"
            )(critic_l1)

        actor_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = scope + "/actor")
        critic_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = scope + "/critic")

        return mu, sigma, q_value, actor_params, critic_params

    def update_global(self, feed_dict): # A worker net updates global net
        self.session.run([self.update_actor, self.update_critic], feed_dict)

    def pull_global(self): # A worker net updates itself according to global net
        self.session.run([self.pull_actor_params, self.pull_critic_params])

    def choose_action(self, state): # A worker net chooses action
        state = state[np.newaxis, :]
        return self.session.run(self.action, feed_dict = { self.state: state })

class Worker(object):
    def __init__(
        self,
        name,
        env_name,
        global_net,
        coordinator,
        session,
        n_feature,
        n_action,
        action_bound,
        actor_optimizer,
        critic_optimizer,
        update_global_iter = 10,
        max_global_epoch = 2000,
        max_epoch_step = 200,
        reward_decay = 0.9,
    ):
        self.env = gym.make(env_name).unwrapped
        self.name = name

        self.AC = ActorCriticNet(
            scope = name,
            global_net = global_net,
            n_feature = n_feature,
            n_action = n_action,
            action_bound = action_bound,
            session = session,
            actor_optimizer = actor_optimizer,
            critic_optimizer = critic_optimizer,
        )
        self.coordinator = coordinator
        self.session = session

        self.update_global_iter = update_global_iter
        self.max_global_epoch = max_global_epoch
        self.max_epoch_step = max_epoch_step
        self.gamma = reward_decay

    def work(self):
        global g_epoch, g_running_reward
        
        total_step = 1
        state_buffer = []
        action_buffer = []
        reward_buffer = []
        while not self.coordinator.should_stop() and g_epoch < self.max_global_epoch:
            state = self.env.reset()
            epoch_reward = 0
            for epoch in range(self.max_epoch_step):
                action = self.AC.choose_action(state)
                next_state, reward, done, info = self.env.step(action)
                done = True if epoch == self.max_epoch_step - 1 else False

                epoch_reward += reward
                state_buffer.append(state)
                action_buffer.append(action)
                reward_buffer.append((reward + 8) / 8) # Normalize

                if total_step % self.update_global_iter == 0 or done: # Update global and assign to local net
                    if done:
                        next_q_value = 0 # Terminal
                    else:
                        next_q_value = self.session.run(self.AC.q_value, feed_dict = {
                            self.AC.state: next_state[np.newaxis, :]
                        })[0, 0]
                    
                    q_value_buffer = []
                    for reward in reward_buffer[:: -1]: # Reverse buffer
                        next_q_value = reward + self.gamma * next_q_value
                        q_value_buffer.append(next_q_value)
                    q_value_buffer.reverse()

                    self.AC.update_global(feed_dict = {
                        self.AC.state: np.vstack(state_buffer),
                        self.AC.action_history: np.vstack(action_buffer),
                        self.AC.q_value_target: np.vstack(q_value_buffer),
                    })

                    state_buffer = []
                    action_buffer = []
                    reward_buffer = []
                    self.AC.pull_global()

                state = next_state
                total_step += 1
                if done:
                    if len(g_running_reward) == 0: # Record running episode reward
                        g_running_reward.append(epoch_reward)
                    else:
                        g_running_reward.append(
                            0.9 * g_running_reward[-1] +
                            0.1 * epoch_reward
                        )

                    print("Name: %s, Ep: %d, R: %d" % (
                        self.name, g_epoch, g_running_reward[-1]
                    ))
                    g_epoch += 1
                    break