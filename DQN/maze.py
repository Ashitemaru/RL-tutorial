"""
Reinforcement learning maze example.
Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].
This script is the environment part of this example.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Updated by Ashitemaru, 2022/07/05.
"""

import numpy as np
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

UNIT = 40  # Pixels
MAZE_H = 4 # Grid height
MAZE_W = 4 # Grid width

class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ["u", "d", "l", "r"]
        self.n_actions = len(self.action_space)
        self.n_features = 2 # How many features it will take to describe a state
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

        # Hell
        hell_center = origin + np.array([UNIT * 2, UNIT])
        self.hell = self.canvas.create_rectangle(
            hell_center[0] - 15, hell_center[1] - 15,
            hell_center[0] + 15, hell_center[1] + 15,
            fill = "black"
        )

        # Oval
        oval_center = origin + UNIT * 2
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill = "yellow"
        )

        # Red rectangle
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill = "red"
        )

        # Pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.1)
        self.canvas.delete(self.rect)

        origin = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill = "red"
        )

        # Return state
        return (
            np.array(self.canvas.coords(self.rect)[: 2]) -
            np.array(self.canvas.coords(self.oval)[: 2])
        ) / (MAZE_H * UNIT)

    def step(self, action):
        state = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])

        if action == 0: # Up
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

        self.canvas.move(self.rect, base_action[0], base_action[1]) # Move
        next_coords = self.canvas.coords(self.rect) # Next state

        # Reward function
        if next_coords == self.canvas.coords(self.oval):
            reward = 1
            done = True
        elif next_coords in [self.canvas.coords(self.hell)]:
            reward = -1
            done = True
        else:
            reward = 0
            done = False
        next_state = (
            np.array(next_coords[: 2]) -
            np.array(self.canvas.coords(self.oval)[: 2])
        ) / (MAZE_H * UNIT)

        return next_state, reward, done

    def render(self):
        # time.sleep(0.1) # Comment it to boost the speed of training model
        self.update()