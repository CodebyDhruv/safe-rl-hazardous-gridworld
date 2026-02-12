import numpy as np
from collections import defaultdict
import pickle
import os

from env.safe_minigrid import SafeMiniGridEnv

def zero_action_values(n_actions):
    return np.zeros(n_actions)

def get_state(env):
    x, y = env.agent_pos
    return (int(x), int(y), int(env.agent_dir))


def epsilon_greedy(Q, state, epsilon, env):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    return np.argmax(Q[state])


def train_q_learning(
    episodes=1000,
    alpha=0.1,
    gamma=0.99,
    epsilon=1.0,
    epsilon_decay=0.995,
    min_epsilon=0.1,
    seed=42
):

    np.random.seed(seed)

    env = SafeMiniGridEnv()
    env.reset(seed=seed)

    Q = defaultdict(lambda: zero_action_values(env.action_space.n))

    episode_rewards = []
    episode_violations = []
    episode_steps = []

    print(f"\n{'='*60}")
    print(f"Training Q-Learning (seed={seed})")
    print(f"{'='*60}\n")

    for ep in range(episodes):
        env.reset()
        state = get_state(env)

        total_reward = 0
        total_violations = 0
        steps = 0
        done = False

        while not done:
            action = epsilon_greedy(Q, state, epsilon, env)

            _, reward, terminated, truncated, info = env.step(action)
            next_state = get_state(env)

            Q[state][action] += alpha * (
                reward + gamma * np.max(Q[next_state]) - Q[state][action]
            )

            state = next_state
            total_reward += reward
            total_violations += info["violation"]
            steps += 1

            done = terminated or truncated

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        episode_rewards.append(total_reward)
        episode_violations.append(total_violations)
        episode_steps.append(steps)

        if ep % 50 == 0 or ep == episodes - 1:
            print(
                f"Episode {ep:4d} | "
                f"Reward: {total_reward:6.2f} | "
                f"Violations: {total_violations:3d} | "
                f"Steps: {steps:3d} | "
                f"Epsilon: {epsilon:.3f}"
            )

    env.close()

    os.makedirs("results", exist_ok=True)
    with open("results/q_learning_Q.pkl", "wb") as f:
        pickle.dump(dict(Q), f)
    
    metrics = {
    "rewards": episode_rewards,
    "violations": episode_violations,
    "steps": episode_steps
}

    with open("results/q_learning_metrics.pkl", "wb") as f:
     pickle.dump(metrics, f)

    print("\nQ-table saved to results/q_learning_Q.pkl")
    print(" - Metrics → results/q_learning_metrics.pkl")

    return Q, episode_rewards, episode_violations, episode_steps


if __name__ == "__main__":
    train_q_learning(episodes=5000, seed=42)
