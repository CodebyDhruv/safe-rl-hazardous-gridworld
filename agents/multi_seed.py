import os
import pickle
import numpy as np

from agents.q_learning import train_q_learning
from agents.sarsa import train_sarsa
from agents.lagrangian_fixed import train_lagrangian_fixed
from agents.lagrangian_adaptive import train_lagrangian_adaptive
from agents.shielding import train_shielded_q_learning


SEEDS = [0, 1, 2, 3, 4]   
EPISODES = 5000


def run_experiment(train_fn, name):
    print("\n" + "="*70)
    print(f"Running Multi-Seed for {name}")
    print("="*70)

    all_rewards = []
    all_violations = []
    all_steps = []

    for seed in SEEDS:
        print(f"\n---- Seed {seed} ----\n")

        result = train_fn(episodes=EPISODES, seed=seed)

        rewards = result[1]
        violations = result[2]

        if len(result) >= 4:
            steps = result[3]
        else:
            steps = None

        all_rewards.append(rewards)
        all_violations.append(violations)
        if steps is not None:
            all_steps.append(steps)

    metrics = {
        "rewards": np.array(all_rewards),
        "violations": np.array(all_violations),
        "steps": np.array(all_steps) if all_steps else None,
        "seeds": SEEDS
    }

    os.makedirs("results_multiseed", exist_ok=True)

    with open(f"results_multiseed/{name}_multiseed.pkl", "wb") as f:
        pickle.dump(metrics, f)

    print(f"\nSaved → results_multiseed/{name}_multiseed.pkl")


if __name__ == "__main__":

    run_experiment(train_q_learning, "q_learning")
    run_experiment(train_sarsa, "sarsa")
    run_experiment(train_lagrangian_fixed, "lagrangian_fixed")
    run_experiment(train_lagrangian_adaptive, "lagrangian_adaptive")
    run_experiment(train_shielded_q_learning, "shielded")
