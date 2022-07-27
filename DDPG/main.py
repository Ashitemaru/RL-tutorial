import gym
import numpy as np
from ddpg import DDPG
import matplotlib.pyplot as plt

MAX_EPISODES = 200
MAX_EPOCH_STEPS = 200
RENDER = False
ENV_NAME = "Pendulum-v0"
MEMORY_CAPACITY = 10000

action_clip_var = 3 # Use to control the exploration

env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)

n_feature = env.observation_space.shape[0]
n_action = env.action_space.shape[0]
action_bound = env.action_space.high

RL = DDPG(
    n_action = n_action,
    n_feature = n_feature,
    action_bound = action_bound,
)

reward_list = []

for i_episode in range(MAX_EPISODES):
    state = env.reset()
    epoch_reward = 0

    for i_step in range(MAX_EPOCH_STEPS):
        if RENDER:
            env.render()

        # Add exploration noise when get the action
        action = RL.choose_action(state)
        action = np.clip(np.random.normal(action, action_clip_var), -2, 2)

        next_state, reward, done, info = env.step(action)
        RL.store_transition(state, action, reward / 10, next_state)

        if RL.memory_ptr > MEMORY_CAPACITY:
            action_clip_var *= .9995 # Decay the action randomness
            RL.learn()

        state = next_state
        epoch_reward += reward

        if i_step == MAX_EPOCH_STEPS - 1:
            print("Epoch: %d, Reward: %d, Explore clip var: %.2f" % (
                i_episode, int(epoch_reward), action_clip_var
            ))

            if epoch_reward > -300:
                RENDER = False

            reward_list.append(int(epoch_reward))
            break

plt.figure()
plt.xlabel("Epoch")
plt.ylabel("Reward")
plt.plot(reward_list)
plt.savefig("./image/reward_plt.png")
plt.show()