# 🛡️ Safe RL Evaluator: Project Workflow & Theoretical Guide

## 🎯 1. Project Aim
The core objective of this project is to develop a robust, statistically validated framework for **Safe Reinforcement Learning**. In many real-world scenarios (robotics, medicine, autonomous driving), maximizing a reward is not enough—the agent must also strictly adhere to safety constraints. This project implements, compares, and visualizes 5 different approaches to solving **Constrained Markov Decision Processes (CMDPs)**.

---

## 🏛️ 2. Theoretical Framework

### A. Core Concepts
*   **CMDP (Constrained Markov Decision Process):** Unlike standard RL which only has Reward ($R$), a CMDP adds a Cost function ($C$). The goal is to maximize $R$ such that $C$ stays below a certain threshold $\delta$.
*   **Tabular Q-Learning:** We use a table to store the "value" of taking an action in a state. This is highly interpretable compared to a black-box neural network.
*   **Hazard Environment:** A 16x16 GridWorld where certain tiles are fatal "hazards". The agent must reach a Goal tile without stepping on red Hazards.

### B. The 5 Algorithmic Archetypes
1.  **Q-Learning (Baseline):** Ignores safety. It only cares about the goal. Useful to show "bad" behavior.
2.  **SARSA (Baseline):** An on-policy learner. Also ignores safety, used as a secondary baseline.
3.  **Fixed Lagrangian:** Adds a penalty to the reward whenever a violation occurs ($R_{new} = R - \lambda \cdot C$). Here, $\lambda$ is a fixed penalty "fine".
4.  **Adaptive Lagrangian:** Instead of a fixed fine, the system dynamically increases the penalty $\lambda$ if violations are too high, and decreases it if the agent is safe. It "learns" how much to care about safety.
5.  **Shielding (Hard Safety):** The "Gold Standard". A physical logic layer sits outside the agent. If the agent tries to move into a hazard, the Shield **overrides** the action and forces a safe one instead.

---

## 🔄 3. How the Pipeline Works (The Flow)

### 1. Training Phase (`agents/multi_seed.py`)
Instead of training once, we train **5 separate agents** for every algorithm using different "Seeds". This ensures our results aren't just "lucky" flukes. This generates `.pkl` files containing the learned brains (Q-tables).

### 2. Aggregation Phase (`evaluation/plots_multiseed.py`)
We take the 5 runs and calculate the **Mean** and **Standard Deviation**.
*   **Shaded Bands:** These in the plots show the variance. Thin bands mean the algorithm is stable; thick ones mean it's unpredictable.

### 3. Meta-Evaluation (`evaluation/auto_evaluator.py`)
A script acts as a "Judge". It looks at the final Reward vs. Violation scores and mathematically decides which algorithm is the "Best" based on your project's specific safety needs.

### 4. Interactive Visualization (`dashboard/`)
The Python data is exported to JSON and rendered in a **React Dashboard**. 
*   **HTML5 Canvas:** We built an engine that replays the agent's path frame-by-frame.
*   **Glassmorphism UI:** A premium design to present results professionally.

---

## 🛠️ 4. Essential Commands

### Environment Activation
```bash
source myenv/bin/activate
```

### Full Pipeline Execution
```bash
# 1. Run multi-seed training (20-40 mins)
python -m agents.multi_seed

# 2. Generate plots and exports
python evaluation/plots_multiseed.py
python -m evaluation.export_replays
python -m evaluation.auto_evaluator

# 3. Update Dashboard
cp -r results dashboard/public/

# 4. Launch Dashboard
cd dashboard && npm run dev
```

---

## 👨‍🏫 5. How to Explain This to Your Supervisor (The "Elevator Pitch")

**Start with the Problem:**
> "Sir, standard RL is dangerous for hardware because it optimizes for reward at the cost of safety. My project aims to solve this by building a comparative framework for Safe RL using CMDP theory."

**Walk Through the Logic:**
> "I compared three levels of safety: **Baselines** (which fail), **Lagrangian methods** (which learn safety through penalties), and **Hard Shielding** (which enforces safety through logical overrides)."

**Highlight the Rigor:**
> "I didn't just run it once. I used a **Multi-seed Evaluation** to prove statistical significance, showing mean performance with standard deviation bands, just like an academic paper."

**Show the Conclusion:**
> "In the end, while Lagrangian methods learn to be safe, **Shielding** provides the only guaranteed zero-violation performance, making it the mathematically superior choice for safety-critical systems."

**End with the Dashboard:**
> "To prove this works, I built a custom **React Dashboard** that visualizes the agent's path in real-time and manages all our experimental data in one professional interface."
