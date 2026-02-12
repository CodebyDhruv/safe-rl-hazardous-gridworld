import pickle
import numpy as np

from env.safe_minigrid import SafeMiniGridEnv
from agents.q_learning import get_state

MOVEMENT_ACTIONS = (0, 1, 2)


def _best_movement_action(Q, state, n_actions):
    q_default = np.zeros(n_actions)
    q_vals = np.array([Q.get(state, q_default)[a] for a in MOVEMENT_ACTIONS])
    best_idx = np.argmax(q_vals)
    best_idxs = np.where(q_vals == q_vals[best_idx])[0]
    return MOVEMENT_ACTIONS[np.random.choice(best_idxs)]


def test_policy(Q, episodes=5, render=False, movement_only=True):
    env = SafeMiniGridEnv(render_mode="human" if render else None)

    for ep in range(episodes):
        env.reset(seed=ep)
        state = get_state(env)

        done = False
        steps = 0
        total_reward = 0
        total_violations = 0

        print(f"\nTest Episode {ep + 1} started")

        while not done and steps < 400:
            if movement_only:
                action = _best_movement_action(Q, state, env.action_space.n)
            else:
                action = np.argmax(Q.get(state, np.zeros(env.action_space.n)))
            _, reward, terminated, truncated, info = env.step(action)

            state = get_state(env)
            total_reward += reward
            total_violations += info["violation"]
            steps += 1

            done = terminated or truncated

        print(
            f"Reward={total_reward:.2f}, "
            f"Violations={total_violations}, "
            f"Steps={steps}"
        )

    env.close()


if __name__ == "__main__":
    algo = input(
        "Enter the algorithm to test (q / sarsa / l_fixed / l_adaptive / shielded): "
    ).strip().lower()

    if algo == "q":
        path = "results/q_learning_Q.pkl"
    elif algo == "sarsa":
        path = "results/sarsa_Q.pkl"
    elif algo == "l_fixed":
        path = "results/lagrangian_fixed_Q.pkl"
    elif algo == "l_adaptive":
        path = "results/lagrangian_adaptive_Q.pkl"
    elif algo == "shielded":
        path = "results/shielded_q_learning_Q.pkl"
    else:
        print("Invalid algorithm name. Use 'q', 'sarsa', 'l_fixed', 'l_adaptive', or 'shielded'.")
        exit(1)

    print(f">>> Loading Q-table from {path}")
    with open(path, "rb") as f:
        Q = pickle.load(f)

    render = input("Render environment? (y/n): ").strip().lower() == "y"

    print(">>> Testing learned policy")
    test_policy(Q, episodes=1, render=render)
