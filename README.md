# 🛡️ Autonomous Safe RL Evaluator

This repository contains a full-stack, mathematically rigorous framework for solving Constrained Markov Decision Processes (CMDPs) using **Safety-Critical Reinforcement Learning**. 

It evaluates and directly compares **5 distinct algorithmic architectures**, mapping their trade-offs between pure utility optimization (Rewards) and rigid physical bounding (Safety Violations):

1. **Q-Learning** (Baseline / Risk-Taking)
2. **SARSA** (Baseline / Risk-Taking)
3. **Fixed Lagrangian** (Moderate / Risk-Aware)
4. **Adaptive Lagrangian** (Moderate / Dynamic Barrier)
5. **Shielded Q-Learning** (Mature / Strict Absolute Safety)

---

## 🚀 Key Features
- **Multi-Seed Statistical Analysis**: Prevents validation flukes by computing mean performances and Standard Deviation (±1 SD shaded bands) across `N` parallel agent initializations.
- **Autonomous Meta-Evaluator**: A Python-based script (`auto_evaluator.py`) that dynamically ingests the converged metrics and autonomously recommends the mathematically optimal deployment model based on hard-coded physical constraint thresholds.
- **React Positional Replay Dashboard**: A Serverless HTML5 Canvas/React UI that dynamically renders the evaluation logic, parses Q-Table trajectories into `[x,y]` coordinates, and animates the agent moving through the GridWorld mathematically.

---

## 💻 Installation & Setup

### Prerequisites
You must have Python 3.10+ and Node.js (`npm`) installed on your machine.

### 1. Python Environment Setup
#### macOS / Linux
```bash
python3 -m venv myenv
source myenv/bin/activate
pip install minigrid numpy matplotlib
```

#### Windows
```powershell
python -m venv myenv
myenv\Scripts\activate
pip install minigrid numpy matplotlib
```

---

## 🧠 Executing the Machine Learning Pipeline

### Step 1: Execute the Training (The Heavy Lifting)
This computes the mathematical boundaries across all 25 parallel seed environments simultaneously (Takes ~20-45 minutes depending on CPU):
```bash
python -m agents.multi_seed
```

### Step 2: Compile the Visual Matrices & Evaluate
Once training concludes, generate the IEEE-standard deviation plots and serialize the Replay vectors into `.json` formatting for the frontend React engine:
```bash
python evaluation/plots_multiseed.py
python -m evaluation.export_replays
python -m evaluation.auto_evaluator
cp -r results dashboard/public/
```

---

## 🖥️ Launching the Presentation Dashboard
To view your simulation results within the premium evaluation engine:
```bash
cd dashboard
npm install       # Only required once
npm run dev
```
Navigate to the provided localhost URL (e.g., `http://localhost:5173`) to view the interactive replay animations, the auto-evaluator recommendations, and the dynamic Pareto frontier comparisons!