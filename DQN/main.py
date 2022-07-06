from maze import Maze
from dqn import DeepQNetwork

def run_maze():
    step = 0
    for _ in range(300):
        # Initial state
        state = env.reset()

        while True:
            # Refresh environment
            env.render()

            # RL choose action based on state
            action = RL.choose_action(state)

            # RL take action and get next state and reward
            next_state, reward, done = env.step(action)

            RL.store_transition(state, action, reward, next_state)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # Swap state, step forward
            state = next_state

            # Break while loop when end of this episode
            if done:
                break
            
            step += 1

    print("Game over")
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = DeepQNetwork(
        env.n_actions,
        env.n_features,
        learning_rate = 0.01,
        reward_decay = 0.9,
        e_greedy = 0.9,
        replace_target_iter = 200,
        memory_size = 2000,
        output_graph = True
    )

    env.after(100, run_maze)
    env.mainloop()
    RL.plot_cost()
