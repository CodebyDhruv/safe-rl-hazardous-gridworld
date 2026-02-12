import numpy as np
import pickle
from collections import defaultdict

from env.safe_minigrid import SafeMiniGridEnv
from agents.q_learning import get_state


def train_lagrangian_fixed(
    episodes=6000,
    alpha=0.1,
    gamma=0.99,
    epsilon=1.0,
    epsilon_decay=0.995,
    min_epsilon=0.1,
    lambda_penalty=0.5,
    seed=42,
):

    np.random.seed(seed)

    env = SafeMiniGridEnv()
    env.reset(seed=seed)

    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    rewards = []
    violations = []
    steps_list = []

    print("\n" + "=" * 60)
    print(f"Training FIXED-λ Lagrangian Q-Learning (λ = {lambda_penalty})")
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

            shaped_reward = reward - lambda_penalty * cost

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
            
        steps_list.append(steps)
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        rewards.append(total_reward)
        violations.append(total_violations)

        if ep % 50 == 0 or ep == episodes - 1:
            print(
                f"Episode {ep:4d} | "
                f"Reward: {total_reward:6.2f} | "
                f"Violations: {total_violations:3d} | "
                f"Epsilon: {epsilon:.3f}"
            )

    env.close()

    with open("results/lagrangian_fixed_Q.pkl", "wb") as f:
        pickle.dump(dict(Q), f)
    
    metrics = {
    "rewards": rewards,
    "violations": violations,
    "steps": steps_list,
    "lambda": 0.5
}

    with open("results/l_fixed_metrics.pkl", "wb") as f:
     pickle.dump(metrics, f)

    print("\nSaved → results/lagrangian_fixed_Q.pkl")
    print(" - Metrics → results/l_fixed_metrics.pkl")
    return rewards, violations


if __name__ == "__main__":
    train_lagrangian_fixed(lambda_penalty=0.5)
