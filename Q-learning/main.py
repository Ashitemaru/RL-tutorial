"""
Reinforcement learning maze example.
Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].
This script is the main part which controls the update method of this example.
The RL is in qlearning.py.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Updated by Ashitemaru, 2022/07/02.
"""

from maze import Maze
from qlearning import QLearningTable

def update():
    for _ in range(100):
        # Initial state
        state = env.reset()

        while True:
            # Refresh the canvas
            env.render()

            # RL choose action based on state
            action = RL.choose_action(str(state))

            # RL take action and get next state and reward
            next_state, reward, done = env.step(action)

            # RL learn from this transition
            RL.learn(str(state), action, reward, str(next_state))

            # Swap state, move ahead
            state = next_state

            # Break while loop when end of this episode
            if done:
                break

    # End of game
    print("Game over")
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable(actions = list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()