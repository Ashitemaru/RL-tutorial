import gym
import tensorflow.compat.v1 as tf
from ac import Actor, Critic

tf.disable_v2_behavior()
tf.set_random_seed(2)

# Superparameters
OUTPUT_GRAPH = True
DISPLAY_REWARD_THRESHOLD = 200 # Renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 1000 # Maximum time step in one episode
RENDER = True # Rendering wastes time

env = gym.make("CartPole-v0")
env.seed(1)
env = env.unwrapped

session = tf.Session()

actor = Actor(
    session = session,
    n_features = env.observation_space.shape[0],
    n_actions = env.action_space.n,
    lr = 0.001
)
critic = Critic(
    session = session,
    n_features = env.observation_space.shape[0],
    lr = 0.01
) # We need a good teacher, so the teacher should learn faster than the actor

session.run(tf.global_variables_initializer())

if OUTPUT_GRAPH:
    tf.summary.FileWriter("logs/", session.graph)

for i_episode in range(3000):
    state = env.reset()
    step_counter = 0
    tracked_reward_list = []

    while True:
        if RENDER:
            env.render()

        action = actor.choose_action(state)
        next_state, reward, done, info = env.step(action)

        if done:
            reward = -20

        tracked_reward_list.append(reward)

        td_error = critic.learn(state, reward, next_state)
        actor.learn(state, action, td_error)

        state = next_state
        step_counter += 1

        if done or step_counter >= MAX_EP_STEPS:
            epoch_reward_sum = sum(tracked_reward_list)

            if "running_reward" not in globals():
                running_reward = epoch_reward_sum
            else:
                running_reward = running_reward * 0.95 + epoch_reward_sum * 0.05

            if running_reward > DISPLAY_REWARD_THRESHOLD:
                RENDER = True

            print("Episode: ", i_episode, ", Reward: ", int(running_reward), sep = "")
            break