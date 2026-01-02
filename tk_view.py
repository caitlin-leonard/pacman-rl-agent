# tk_view.py
import tkinter as tk
from env_pacman import PacmanEnv, N_ACTIONS

CELL_SIZE = 40
DELAY_MS = 150  # time between steps


class PacmanViewer:
    def __init__(self, root, Q=None):
        self.root = root
        self.env = PacmanEnv()
        self.Q = Q  # can be None: then random actions

        w = self.env.width * CELL_SIZE
        h = self.env.height * CELL_SIZE
        self.canvas = tk.Canvas(root, width=w, height=h, bg="black")
        self.canvas.pack()

        self.state = self.env.reset()
        self.draw()

        self.root.after(DELAY_MS, self.step_loop)

    def draw(self):
        self.canvas.delete("all")

        # Draw map
        for r in range(self.env.height):
            for c in range(self.env.width):
                x0 = c * CELL_SIZE
                y0 = r * CELL_SIZE
                x1 = x0 + CELL_SIZE
                y1 = y0 + CELL_SIZE

                ch = self.env.raw_map[r][c]
                if ch == '#':
                    self.canvas.create_rectangle(x0, y0, x1, y1, fill="navy")

        # Pellets
        for (r, c) in self.env.pellets:
            x = c * CELL_SIZE + CELL_SIZE / 2
            y = r * CELL_SIZE + CELL_SIZE / 2
            self.canvas.create_oval(x-4, y-4, x+4, y+4, fill="white")

        # Ghost
        gr, gc = self.env.ghost_pos
        gx0 = gc * CELL_SIZE + 5
        gy0 = gr * CELL_SIZE + 5
        gx1 = gx0 + CELL_SIZE - 10
        gy1 = gy0 + CELL_SIZE - 10
        self.canvas.create_rectangle(gx0, gy0, gx1, gy1, fill="red")

        # Pacman
        pr, pc = self.env.pacman_pos
        px = pc * CELL_SIZE + CELL_SIZE / 2
        py = pr * CELL_SIZE + CELL_SIZE / 2
        self.canvas.create_oval(px-12, py-12, px+12, py+12, fill="yellow")

    def greedy_action(self, state):
        if self.Q is None:
            import random
            return random.randrange(N_ACTIONS)
        # choose best action according to Q
        qs = [self.Q.get((state, a), 0.0) for a in range(N_ACTIONS)]
        max_q = max(qs)
        best_actions = [a for a, q in enumerate(qs) if q == max_q]
        import random
        return random.choice(best_actions)

    def step_loop(self):
        action = self.greedy_action(self.state)
        next_state, reward, done, _ = self.env.step(action)
        self.state = next_state
        self.draw()

        if done:
            # restart automatically
            self.state = self.env.reset()
        self.root.after(DELAY_MS, self.step_loop)


if __name__ == "__main__":
    import tkinter as tk

    root = tk.Tk()
    root.title("Pacman RL Viewer")

    viewer = PacmanViewer(root, Q=None)  # pass root as first arg
    root.mainloop()
