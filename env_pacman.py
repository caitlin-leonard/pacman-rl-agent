# env_pacman.py
import numpy as np

ACTIONS = {
    0: (-1, 0),   # up
    1: (0, 1),    # right
    2: (1, 0),    # down
    3: (0, -1),   # left
}
N_ACTIONS = len(ACTIONS)


class PacmanEnv:
    def __init__(self):
        # Tiny 7x7 map for tabular Q-learning
        # # = wall, . = pellet, P = pacman start, G = ghost
        self.raw_map = [
            "#######",
            "#P....#",
            "#.###.#",
            "#..G..#",
            "#.###.#",
            "#.....#",
            "#######",
        ]
        self.height = len(self.raw_map)
        self.width = len(self.raw_map[0])

        # Will be filled in reset()
        self.pacman_pos = None
        self.ghost_pos = None
        self.pellets = set()

    def reset(self):
        """Reset environment, return initial state."""
        self.pellets.clear()
        self.pacman_pos = None
        self.ghost_pos = None

        for r in range(self.height):
            for c in range(self.width):
                ch = self.raw_map[r][c]
                if ch == 'P':
                    self.pacman_pos = (r, c)
                elif ch == 'G':
                    self.ghost_pos = (r, c)
                elif ch == '.':
                    self.pellets.add((r, c))

        # Return initial state (for now: just pacman position)
        return self._get_state()

    def _get_state(self):
        """Encode state as a tuple; start simple: just pacman row, col."""
        return self.pacman_pos

    def _is_wall(self, pos):
        r, c = pos
        return self.raw_map[r][c] == '#'

    def step(self, action):
        """
        Take an action (0..3), return (next_state, reward, done, info).
        Reward design (simple to start):
          +1  pellet eaten
          -10 caught by ghost
          -0.01 time step cost
        Episode ends when pacman dies or all pellets eaten.
        """
        dr, dc = ACTIONS[action]
        pr, pc = self.pacman_pos
        nr, nc = pr + dr, pc + dc

        # If next cell is wall, stay in place
        if self._is_wall((nr, nc)):
            nr, nc = pr, pc

        self.pacman_pos = (nr, nc)

        reward = -0.01
        done = False
        info = {}

        # Eat pellet
        if self.pacman_pos in self.pellets:
            self.pellets.remove(self.pacman_pos)
            reward += 1.0

        # Simple ghost: stays fixed for now (you can move it later)
        if self.pacman_pos == self.ghost_pos:
            reward -= 10.0
            done = True

        # All pellets eaten -> win
        if not self.pellets:
            reward += 10.0
            done = True

        next_state = self._get_state()
        return next_state, reward, done, info

    def render_ansi(self):
        """Text render in terminal for debugging."""
        grid = [list(row) for row in self.raw_map]
        for r, c in self.pellets:
            grid[r][c] = '.'

        gr, gc = self.ghost_pos
        grid[gr][gc] = 'G'

        pr, pc = self.pacman_pos
        grid[pr][pc] = 'P'

        print("\n".join("".join(row) for row in grid))
        print()
