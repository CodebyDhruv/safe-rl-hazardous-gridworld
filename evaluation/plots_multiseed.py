import os
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
    if len(x) < window: return x
    return np.convolve(x, np.ones(window)/window, mode="valid")

def main():
    print("="*70)
    print("GENERATING MULTI-SEED PLOTS (ORGANIZED PER ALGORITHM)")
    print("="*70)

    final_rewards = []
    final_violations = []
    labels = []

    for algo in ALGORITHMS:
        out_dir = f"results/plots/{algo}"
        os.makedirs(out_dir, exist_ok=True)
        
        path = f"results_multiseed/{algo}_multiseed.pkl"
        if not os.path.exists(path):
            print(f"Skipping {algo}: No data found.")
            continue

        with open(path, "rb") as f:
            data = pickle.load(f)

        rewards = data["rewards"]        
        violations = data["violations"]
        steps = data.get("steps", None)

        mean_reward = np.mean(rewards, axis=0)
        std_reward = np.std(rewards, axis=0)
        
        mean_violation = np.mean(violations, axis=0)
        std_violation = np.std(violations, axis=0)

        # 1. Reward Trade-off
        plt.figure()
        sm_rew = moving_average(mean_reward)
        sm_std_rew = moving_average(std_reward)
        x = np.arange(len(sm_rew))
        plt.plot(x, sm_rew, label="Mean Reward", color="blue")
        plt.fill_between(x, sm_rew - sm_std_rew, sm_rew + sm_std_rew, color="blue", alpha=0.2)
        plt.title(f"{algo.replace('_', ' ').title()} - Reward vs Episodes")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        plt.savefig(f"{out_dir}/{algo}_reward_tradeoff.png", bbox_inches='tight')
        plt.close()

        # 2. Violation Rate
        plt.figure()
        if steps is not None:
            # Prevent division by zero
            safe_steps = np.maximum(steps, 1)
            v_rate = violations / safe_steps
            mean_v_rate = np.mean(v_rate, axis=0)
            std_v_rate = np.std(v_rate, axis=0)
            
            sm_v_rate = moving_average(mean_v_rate)
            sm_std_v_rate = moving_average(std_v_rate)
            x_v = np.arange(len(sm_v_rate))
            plt.plot(x_v, sm_v_rate, label="Mean V-Rate", color="red")
            plt.fill_between(x_v, sm_v_rate - sm_std_v_rate, sm_v_rate + sm_std_v_rate, color="red", alpha=0.2)
        else:
            sm_v = moving_average(mean_violation)
            sm_std_v = moving_average(std_violation)
            x_v = np.arange(len(sm_v))
            plt.plot(x_v, sm_v, label="Mean Violations", color="red")
            plt.fill_between(x_v, sm_v - sm_std_v, sm_v + sm_std_v, color="red", alpha=0.2)
        
        plt.title(f"{algo.replace('_', ' ').title()} - Violation Rate")
        plt.xlabel("Episode")
        plt.ylabel("Violation Rate")
        plt.legend()
        plt.savefig(f"{out_dir}/{algo}_violation_rate.png", bbox_inches='tight')
        plt.close()

        # Save for Pareto across all
        final_rewards.append(np.mean(mean_reward[-50:]))
        final_violations.append(np.mean(mean_violation[-50:]))
        labels.append(algo)

    # 3. Pareto Analysis (Performance vs Safety)
    os.makedirs("results/plots/comparisons", exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.scatter(final_violations, final_rewards, color='purple', s=100)
    for i, label in enumerate(labels):
        plt.annotate(label, (final_violations[i], final_rewards[i]), textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.title("Pareto Analysis: Performance vs Safety (All Agents Avg)")
    plt.xlabel("Average Violations (Lower is Better)")
    plt.ylabel("Average Reward (Higher is Better)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig("results/plots/comparisons/pareto_analysis.png", bbox_inches='tight')
    plt.close()
    
    print("✅ Successfully generated specialized plots and nested them into individual algorithm folders!")

if __name__ == "__main__":
    main()
