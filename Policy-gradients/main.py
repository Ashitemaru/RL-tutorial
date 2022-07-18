"""
Policy Gradient, Reinforcement Learning.
The cart pole example
View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Updated by Ashitemaru, 2022/07/18.
"""

import gym
from pg import PolicyGradient
import matplotlib.pyplot as plt

DISPLAY_REWARD_THRESHOLD = -2000 # Renders environment if total episode reward is greater then this threshold
RENDER = False # Rendering wastes time

env = gym.make("MountainCar-v0")
env.seed(1)
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = PolicyGradient(
    n_actions = env.action_space.n,
    n_features = env.observation_space.shape[0],
    learning_rate = 0.03,
    reward_decay = 0.99,
    output_graph = True,
)

for i_episode in range(3000):
    state = env.reset()

    while True:
        if RENDER:
            env.render()

        action = RL.choose_action(state)
        next_state, reward, done, info = env.step(action)
        RL.store_transition(state, action, reward)

        if done:
            epoch_reward_sum = sum(RL.epoch_reward_list)

            if "running_reward" not in globals():
                running_reward = epoch_reward_sum
            else:
                running_reward = running_reward * 0.99 + epoch_reward_sum * 0.01

            if running_reward > DISPLAY_REWARD_THRESHOLD:
                RENDER = True

            print("Episode: ", i_episode, ", Reward: ", int(running_reward), sep = "")

            discounted_epoch_reward_list = RL.learn()

            if i_episode == 30:
                plt.plot(discounted_epoch_reward_list)
                plt.xlabel("Episode steps")
                plt.ylabel("Normalized state-action value")
                plt.show()
            break

        state = next_state