"""
Updated by Ashitemaru, 2022/07/02.
"""

from maze import Maze
from model import SarsaTable, SarsaLambdaTable

USE_LAMBDA = True

def update():
    for epoch in range(100):
        # Initial state
        state = env.reset()

        print(f"Start epoch: {epoch}")

        if USE_LAMBDA: # Clear the E table
            RL.e_table *= 0

        # RL choose action based on state
        action = RL.choose_action(str(state))

        while True:
            # Refresh canvas
            env.render()

            # RL take action and get next state and reward
            next_state, reward, done = env.step(action)

            # RL choose action based on next state
            next_action = RL.choose_action(str(next_state))

            # RL learn from this transition
            RL.learn(str(state), action, reward, str(next_state), next_action)

            # Swap state and action
            state = next_state
            action = next_action

            # Break while loop when end of this episode
            if done:
                break

    # End of game
    print("Game over")
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    if USE_LAMBDA:
        RL = SarsaLambdaTable(actions = list(range(env.n_actions)))
    else:
        RL = SarsaTable(actions = list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()