import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


PLOTS_DIR = "results/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


def moving_average(x, window=50):
    return np.convolve(x, np.ones(window)/window, mode='valid')


def load_metrics(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def plot_rewards(metrics, algo_name):
    rewards = metrics["rewards"]
    smoothed = moving_average(rewards)

    plt.figure(figsize=(8, 5))
    plt.plot(smoothed)
    plt.title(f"{algo_name} - Rewards (Smoothed)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid()
    save_path = f"{PLOTS_DIR}/{algo_name}_rewards.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_violations(metrics, algo_name):
    violations = metrics["violations"]
    smoothed = moving_average(violations)

    plt.figure(figsize=(8, 5))
    plt.plot(smoothed)
    plt.title(f"{algo_name} - Violations (Smoothed)")
    plt.xlabel("Episode")
    plt.ylabel("Violations")
    plt.grid()
    save_path = f"{PLOTS_DIR}/{algo_name}_violations.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_violation_rate(metrics, algo_name):
    violations = np.array(metrics["violations"])
    steps = np.array(metrics["steps"])

    rate = violations / np.maximum(steps, 1)
    smoothed = moving_average(rate)

    plt.figure(figsize=(8, 5))
    plt.plot(smoothed)
    plt.title(f"{algo_name} - Violation Rate")
    plt.xlabel("Episode")
    plt.ylabel("Violation Rate")
    plt.grid()
    save_path = f"{PLOTS_DIR}/{algo_name}_violation_rate.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_performance_vs_safety(metrics, algo_name):
    rewards = metrics["rewards"]
    violations = metrics["violations"]

    plt.figure(figsize=(6, 6))
    plt.scatter(violations, rewards, alpha=0.4)
    plt.title(f"{algo_name} - Performance vs Safety")
    plt.xlabel("Violations")
    plt.ylabel("Reward")
    plt.grid()
    save_path = f"{PLOTS_DIR}/{algo_name}_pareto.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def generate_all_plots(metrics_path, algo_name):
    print(f"\nGenerating plots for {algo_name}...")
    metrics = load_metrics(metrics_path)

    plot_rewards(metrics, algo_name)
    plot_violations(metrics, algo_name)
    plot_violation_rate(metrics, algo_name)
    plot_performance_vs_safety(metrics, algo_name)


if __name__ == "__main__":
    generate_all_plots("results/q_learning_metrics.pkl", "q_learning")
    generate_all_plots("results/sarsa_metrics.pkl", "sarsa")
    generate_all_plots("results/l_fixed_metrics.pkl", "lagrangian_fixed")
    generate_all_plots("results/l_adaptive_metrics.pkl", "lagrangian_adaptive")
    generate_all_plots("results/shielding_metrics.pkl", "shielded")
