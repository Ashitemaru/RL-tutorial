import multiprocessing
import threading
import tensorflow.compat.v1 as tf
import numpy as np
import gym
import os
import shutil
import matplotlib.pyplot as plt
from a3c import ActorCriticNet, Worker, g_running_reward

tf.disable_v2_behavior()

np.random.seed(2)
tf.set_random_seed(2)

session = tf.Session()
env = gym.make("Pendulum-v0")

ACTOR_LR = 0.0001 # Learning rate for actor
CRITIC_LR = 0.001 # Learning rate for critic
OUTPUT_GRAPH = True

N_FEATURE = env.observation_space.shape[0]
N_ACTION = env.action_space.shape[0]
ACTION_BOUND = [env.action_space.low, env.action_space.high]

with tf.device("/cpu:0"):
    # Create global AC net
    actor_optimizer = tf.train.RMSPropOptimizer(ACTOR_LR, name = "actor_opt")
    critic_optimizer = tf.train.RMSPropOptimizer(CRITIC_LR, name = "critic_opt")

    global_net = ActorCriticNet(
        scope = ActorCriticNet.GLOBAL_NET_SCOPE,
        n_feature = N_FEATURE,
        n_action = N_ACTION,
        action_bound = ACTION_BOUND,
        session = session,
        actor_optimizer = actor_optimizer,
        critic_optimizer = critic_optimizer,
    )

    coordinator = tf.train.Coordinator()
    
    # Create workers
    worker_list = []
    n_workers = multiprocessing.cpu_count()
    print(f"Using workers: {n_workers}")

    for i in range(n_workers):
        worker_list.append(Worker(
            name = f"worker_{i}",
            env_name = "Pendulum-v0",
            global_net = global_net,
            coordinator = coordinator,
            session = session,
            n_action = N_ACTION,
            n_feature = N_FEATURE,
            action_bound = ACTION_BOUND,
            actor_optimizer = actor_optimizer,
            critic_optimizer = critic_optimizer,
        ))

    session.run(tf.global_variables_initializer())

    if OUTPUT_GRAPH:
        if os.path.exists("./logs"):
            shutil.rmtree("./logs")
        tf.summary.FileWriter("./logs", session.graph)

    # Start all the workers
    worker_thread_list = []
    for worker in worker_list:
        worker_thread = threading.Thread(target = lambda: worker.work())
        worker_thread.start()
        worker_thread_list.append(worker_thread)
    coordinator.join(worker_thread_list)

    # Plot
    plt.plot(np.arange(len(g_running_reward)), g_running_reward)
    plt.xlabel("Step")
    plt.ylabel("Total moving reward")
    plt.savefig("./image/reward_plt.png")
    plt.show()
    