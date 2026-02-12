🛡️ Safe Reinforcement Learning in GridWorld

A comparative study of Safety-Constrained Reinforcement Learning algorithms implemented on a custom 16×16 GridWorld environment.

This project demonstrates how different RL algorithms behave under safety constraints and analyzes the trade-off between performance (reward) and safety (constraint violations).

📖 Project Motivation

Standard Reinforcement Learning maximizes reward without considering safety constraints.

In real-world systems like:

Autonomous vehicles

Robotics

Healthcare decision systems

Industrial control

Unsafe actions can cause catastrophic failures.

This project explores how to incorporate safety into RL through:

Cost penalties

Lagrangian optimization

Hard action shielding

🧠 Algorithms Implemented

We implemented and compared the following algorithms:

Q-Learning (Baseline)

SARSA

Lagrangian Q-Learning (Fixed λ)

Lagrangian Q-Learning (Adaptive λ)

Hard Shielded Q-Learning

Each algorithm is evaluated on:

Episode Reward

Constraint Violations

Violation Rate

Performance vs Safety trade-off

🏗️ Environment Description

Grid Size: 16×16

State: (x, y, direction)

Actions: Turn Left, Turn Right, Move Forward

Hazards: Predefined unsafe cells

Goal: Reach target location

Constraint: Entering hazard = violation

Reward Structure

Small negative reward per step

Positive reward for reaching goal

Violation penalty (for constrained algorithms)

📊 Evaluation Metrics

For each episode, we track:

Total Reward

Number of Violations

Violation Rate (violations / steps)

Steps to Goal

Shielded Action Count (for shielding)

Lagrange Multiplier λ (for adaptive method)

📈 Key Plots

The following plots are generated:

Reward vs Violations

Violation Rate over Episodes

Performance vs Safety (Pareto-style comparison)

Moving Average Reward

λ Convergence (Adaptive Lagrangian)

Multi-seed Mean ± Std Performance

🧪 Multi-Seed Evaluation

To ensure robustness, each algorithm was trained using multiple random seeds.

Final reported results are:

Mean performance across seeds

Standard deviation across seeds

This prevents overfitting to a single random initialization.

📂 Project Structure
safe_rl/
│
├── env/
│   └── safe_minigrid.py
│
├── agents/
│   ├── q_learning.py
│   ├── sarsa.py
│   ├── lagrangian_fixed.py
│   ├── lagrangian_adaptive.py
│   └── shielding.py
│
├── results/
│   ├── *_Q.pkl
│   ├── *_metrics.pkl
│
├── plots/
│   └── plotting scripts
│
└── README.md

🚀 How to Run
1️⃣ Create Virtual Environment
python -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt

2️⃣ Train Algorithms
python -m agents.q_learning
python -m agents.sarsa
python -m agents.lagrangian_fixed
python -m agents.lagrangian_adaptive
python -m agents.shielding

3️⃣ Plot Results
python plot_results.py

📌 Results Summary
Algorithm	Reward	Violations	Safety Level
Q-Learning	High	High	Unsafe
SARSA	Medium	Medium	Moderately Safe
Fixed Lagrangian	Slightly Lower	Low	Safe
Adaptive Lagrangian	Balanced	Very Low	Safer
Shielding	Lower Reward	0	Fully Safe

(Exact values depend on seed and hazard layout)

🔍 Observations

Standard Q-learning maximizes reward but ignores safety.

SARSA behaves more conservatively.

Lagrangian methods balance reward and safety via penalty tuning.

Hard shielding guarantees zero violations but may reduce performance.

Adaptive λ converges to an optimal safety-performance balance.

🎯 Key Learning Outcomes

Understanding constrained MDPs

Lagrangian relaxation in RL

Hard vs soft constraint enforcement

Multi-seed experimental evaluation

Pareto frontier analysis in RL

📚 Theoretical Background

This project is based on:

Constrained Markov Decision Processes (CMDP)

Lagrangian Relaxation

Safe Reinforcement Learning

Hard Action Shielding