"""
Reinforcement learning maze example.
Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].
This script is the environment part of this example.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Updated by Ashitemaru, 2022/07/02.
"""

import numpy as np
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

UNIT = 40   # Pixels
MAZE_H = 4  # Grid height
MAZE_W = 4  # Grid width

class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ["u", "d", "l", "r"]
        self.n_actions = len(self.action_space)
        self.title("maze")
        self.geometry("{0}x{1}".format(MAZE_H * UNIT, MAZE_H * UNIT))
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(
            self,
            bg = "white",
            height = MAZE_H * UNIT,
            width = MAZE_W * UNIT
        )

        # Create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)

        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # Create origin
        origin = np.array([20, 20])

        # Hell1
        hell1_center = origin + np.array([UNIT * 2, UNIT])
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - 15, hell1_center[1] - 15,
            hell1_center[0] + 15, hell1_center[1] + 15,
            fill = "black"
        )

        # Hell2
        hell2_center = origin + np.array([UNIT, UNIT * 2])
        self.hell2 = self.canvas.create_rectangle(
            hell2_center[0] - 15, hell2_center[1] - 15,
            hell2_center[0] + 15, hell2_center[1] + 15,
            fill = "black"
        )

        # Create oval
        oval_center = origin + UNIT * 2
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill = "yellow"
        )

        # Create red rectangle
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill = "red"
        )

        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect)

        origin = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill = "red"
        )

        return self.canvas.coords(self.rect)

    def step(self, action):
        state = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])

        if action == 0:   # Up
            if state[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1: # Down
            if state[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2: # Right
            if state[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3: # Left
            if state[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(self.rect, base_action[0], base_action[1]) # Move agent
        next_state = self.canvas.coords(self.rect) # Get next state

        # Calc the reward function
        if next_state == self.canvas.coords(self.oval):
            reward = 1
            done = True
            next_state = "terminal"
        elif next_state in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2)]:
            reward = -1
            done = True
            next_state = "terminal"
        else:
            reward = 0
            done = False

        return next_state, reward, done

    def render(self):
        time.sleep(0.1)
        self.update()

if __name__ == "__main__":
    pass