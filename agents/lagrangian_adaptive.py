import numpy as np
import pickle
from collections import defaultdict
import os

from env.safe_minigrid import SafeMiniGridEnv
from agents.q_learning import get_state


def train_lagrangian_adaptive(
    episodes=8000,
    alpha=0.1,
    gamma=0.99,
    epsilon=1.0,
    epsilon_decay=0.995,
    min_epsilon=0.1,
    lambda_init=0.2,
    lambda_lr=0.5,
    target_violation_rate=0.02,
    lambda_max=5.0,
    seed=42,
):

    np.random.seed(seed)

    env = SafeMiniGridEnv()
    env.reset(seed=seed)

    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    lambda_val = lambda_init

    rewards = []
    violations = []
    steps_list = []
    lambda_history = []

    print("\n" + "=" * 60)
    print("Training ADAPTIVE-λ Lagrangian Q-Learning")
    print("=" * 60 + "\n")

    for ep in range(episodes):
        env.reset()
        state = get_state(env)

        total_reward = 0.0
        total_violations = 0
        steps = 0
        done = False

        while not done and steps < 400:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])

            _, reward, terminated, truncated, info = env.step(action)
            next_state = get_state(env)

            cost = info["violation"]

            shaped_reward = reward - lambda_val * cost

            Q[state][action] += alpha * (
                shaped_reward
                + gamma * np.max(Q[next_state])
                - Q[state][action]
            )

            state = next_state
            total_reward += reward
            total_violations += cost
            steps += 1

            done = terminated or truncated

        violation_rate = total_violations / max(1, steps)

        lambda_val = np.clip(
            lambda_val + lambda_lr * (violation_rate - target_violation_rate),
            0.0,
            lambda_max,
        )

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        rewards.append(total_reward)
        violations.append(total_violations)
        steps_list.append(steps)
        lambda_history.append(lambda_val)

        if ep % 100 == 0 or ep == episodes - 1:
            print(
                f"Episode {ep:4d} | "
                f"Reward: {total_reward:6.2f} | "
                f"Violations: {total_violations:3d} | "
                f"Steps: {steps:3d} | "
                f"λ: {lambda_val:.3f}"
            )

    env.close()

    os.makedirs("results", exist_ok=True)

    with open("results/lagrangian_adaptive_Q.pkl", "wb") as f:
        pickle.dump(dict(Q), f)

    metrics = {
        "rewards": rewards,
        "violations": violations,
        "steps": steps_list,
        "lambda_history": lambda_history,
    }

    with open("results/l_adaptive_metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)

    print("\nSaved → results/lagrangian_adaptive_Q.pkl")
    print("Saved → results/l_adaptive_metrics.pkl")

    return rewards, violations, lambda_history


if __name__ == "__main__":
    train_lagrangian_adaptive()
