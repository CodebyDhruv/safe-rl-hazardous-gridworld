import pickle
import numpy as np
import matplotlib.pyplot as plt


ALGORITHMS = [
    "q_learning",
    "sarsa",
    "lagrangian_fixed",
    "lagrangian_adaptive",
    "shielded"
]


def moving_average(x, window=50):
    return np.convolve(x, np.ones(window)/window, mode="valid")


for algo in ALGORITHMS:

    print("\n" + "="*70)
    print(f"{algo.upper()} MULTI-SEED RESULTS")
    print("="*70)

    with open(f"results_multiseed/{algo}_multiseed.pkl", "rb") as f:
        data = pickle.load(f)

    rewards = data["rewards"]        
    violations = data["violations"]

    mean_reward = np.mean(rewards, axis=0)
    std_reward = np.std(rewards, axis=0)

    mean_violation = np.mean(violations, axis=0)

    print(f"Final Reward: {mean_reward[-100:].mean():.3f} ± {std_reward[-100:].mean():.3f}")
    print(f"Final Violations: {mean_violation[-100:].mean():.3f}")


    plt.figure()
    plt.plot(moving_average(mean_reward))
    plt.title(f"{algo} - Mean Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")


    plt.figure()
    plt.plot(moving_average(mean_violation))
    plt.title(f"{algo} - Mean Violations")
    plt.xlabel("Episode")
    plt.ylabel("Violations")

plt.show()
