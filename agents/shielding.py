import numpy as np
from collections import defaultdict
import pickle
import os

from env.safe_minigrid import SafeMiniGridEnv
from agents.q_learning import get_state


def predict_next_position(env, action):
    x, y = env.agent_pos
    d = int(env.agent_dir)

    if action == 0:
        return (int(x), int(y)), (d - 1) % 4

    elif action == 1:
        return (int(x), int(y)), (d + 1) % 4

    elif action == 2:
        dx, dy = [(1, 0), (0, 1), (-1, 0), (0, -1)][d]
        return (int(x) + dx, int(y) + dy), d

    return (int(x), int(y)), d


def is_safe_action(env, action):
    if action != 2:
        return True

    next_pos, _ = predict_next_position(env, action)

    if not (0 <= next_pos[0] < env.grid.width and
            0 <= next_pos[1] < env.grid.height):
        return False

    if next_pos in env.hazards:
        return False

    cell = env.grid.get(next_pos[0], next_pos[1])
    if cell is not None and not cell.can_overlap():
        return False

    return True

MOVEMENT_ACTIONS = (0, 1, 2)

def select_shielded_action(Q, state, epsilon, env):
    if np.random.rand() < epsilon:
        candidate = np.random.choice(MOVEMENT_ACTIONS)
    else:
        q_movement = np.array([Q[state][a] for a in MOVEMENT_ACTIONS])
        best_idx = np.argmax(q_movement)
        best_idxs = np.where(q_movement == q_movement[best_idx])[0]
        candidate = MOVEMENT_ACTIONS[np.random.choice(best_idxs)]

    if is_safe_action(env, candidate):
        return candidate, 0

    safe_actions = [a for a in MOVEMENT_ACTIONS if is_safe_action(env, a)]

    if safe_actions:
        q_vals = np.array([Q[state][a] for a in safe_actions])
        best_idx = np.argmax(q_vals)
        best_idxs = np.where(q_vals == q_vals[best_idx])[0]
        idx = np.random.choice(best_idxs)
        return safe_actions[idx], 1

    
    return np.random.choice([0, 1]), 1

def train_shielded_q_learning(
    episodes=1000,
    alpha=0.1,
    gamma=0.99,
    epsilon=1.0,
    epsilon_decay=0.9975,
    min_epsilon=0.05,
    seed=42
):
    np.random.seed(seed)

    env = SafeMiniGridEnv()
    env.reset(seed=seed)

    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    rewards, violations, steps_list, blocked_list = [], [], [], []


    print("\n" + "="*60)
    print("Training SHIELDED Q-Learning (Hard Safety)")
    print("="*60 + "\n")

    for ep in range(episodes):
        env.reset()
        state = get_state(env)

        total_reward = 0
        total_violations = 0
        steps = 0
        blocked = 0
        done = False

        while not done:
            action, was_blocked = select_shielded_action(Q, state, epsilon, env)

            _, reward, terminated, truncated, info = env.step(action)
            next_state = get_state(env)
            next_q_max = np.max([Q[next_state][a] for a in MOVEMENT_ACTIONS])
            Q[state][action] += alpha * (
                reward + gamma * next_q_max - Q[state][action]
            )

            state = next_state
            total_reward += reward
            total_violations += info["violation"]
            blocked += was_blocked
            steps += 1

            done = terminated or truncated

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        rewards.append(total_reward)
        violations.append(total_violations)
        steps_list.append(steps)
        blocked_list.append(blocked)
        
        if ep % 50 == 0 or ep == episodes - 1:
            print(
                f"Episode {ep:4d} | "
                f"Reward: {total_reward:6.2f} | "
                f"Violations: {total_violations:2d} | "
                f"Steps: {steps:3d} | "
                f"Blocked: {blocked:3d} | "
                f"Epsilon: {epsilon:.3f}"
            )

    env.close()

    os.makedirs("results", exist_ok=True)
    with open("results/shielded_q_learning_Q.pkl", "wb") as f:
        pickle.dump(dict(Q), f)
    
    metrics = {
    "rewards": rewards,
    "violations": violations,
    "steps": steps_list,
    "blocked": blocked_list
}

    with open("results/shielding_metrics.pkl", "wb") as f:
     pickle.dump(metrics, f)

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print("Last 100 episodes average:")
    print(f"  Reward:     {np.mean(rewards[-100:]):6.2f}")
    print(f"  Violations: {np.mean(violations[-100:]):6.2f}")
    print(f"  Steps:      {np.mean(steps_list[-100:]):6.2f}")
    print(f"  Blocked:    {np.mean(blocked_list[-100:]):6.2f}")
    print("\nSaved → results/shielded_q_learning_Q.pkl")
    print(" - Metrics → results/shielding_metrics.pkl")
    return Q, rewards, violations, steps_list, blocked_list

if __name__ == "__main__":
    train_shielded_q_learning(episodes=5000, seed=42)
