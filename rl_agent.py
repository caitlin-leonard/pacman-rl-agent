# rl_agent.py
import random
from collections import defaultdict

from env_pacman import PacmanEnv, N_ACTIONS

# Hyperparameters
ALPHA = 0.5          # learning rate
GAMMA = 0.99         # discount factor
EPSILON_START = 1.0  # epsilon-greedy
EPSILON_END = 0.05
EPSILON_DECAY_EPISODES = 500
N_EPISODES = 1000


def epsilon_by_episode(episode):
    if episode >= EPSILON_DECAY_EPISODES:
        return EPSILON_END
    frac = episode / EPSILON_DECAY_EPISODES
    return EPSILON_START + frac * (EPSILON_END - EPSILON_START)


def choose_action(Q, state, epsilon):
    if random.random() < epsilon:
        return random.randrange(N_ACTIONS)
    # greedy
    qs = [Q[(state, a)] for a in range(N_ACTIONS)]
    max_q = max(qs)
    # break ties randomly
    best_actions = [a for a, q in enumerate(qs) if q == max_q]
    return random.choice(best_actions)


def train():
    env = PacmanEnv()
    Q = defaultdict(float)

    for ep in range(N_EPISODES):
        state = env.reset()
        epsilon = epsilon_by_episode(ep)
        total_reward = 0.0
        steps = 0

        while True:
            action = choose_action(Q, state, epsilon)
            next_state, reward, done, _ = env.step(action)

            # Q-learning update
            best_next = max(Q[(next_state, a)] for a in range(N_ACTIONS))
            old_q = Q[(state, action)]
            Q[(state, action)] = old_q + ALPHA * (reward + GAMMA * best_next - old_q)

            state = next_state
            total_reward += reward
            steps += 1

            if done or steps > 500:
                break

        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}/{N_EPISODES}, "
                  f"epsilon={epsilon:.3f}, total_reward={total_reward:.2f}, steps={steps}")

    return Q


if __name__ == "__main__":
    Q = train()
    print("Training finished.")
