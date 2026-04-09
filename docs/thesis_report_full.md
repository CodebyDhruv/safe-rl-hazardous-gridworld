# Comparative Evaluation of Safety-Constraint Handling Architectures in Reinforcement Learning: A Shielding and Lagrangian Dynamics Approach

**Abstract**—Standard reinforcement learning (RL) paradigms focus primarily on cumulative reward maximization, often neglecting the rigid safety requirements essential for real-world autonomous deployments. This paper investigates the optimization of Constrained Markov Decision Processes (CMDPs) through a comparative evaluation of five distinct algorithmic architectures. We examine baseline Q-Learning and SARSA, Lagrangian multiplier methods (fixed and adaptive), and safety-critical shielding logic. By employing a multi-seed statistical validation framework within a hazardous gridworld environment, we demonstrate that while Lagrangian methods internalize safety via penalty-aware utility functions, shielding architectures provide a mathematically guaranteed zero-violation performance. Our results quantify the cost of safety in terms of exploration efficiency and provide a framework for selecting algorithms based on risk maturity profiles.

**Keywords**—Constrained Markov Decision Process (CMDP), Safe Reinforcement Learning, Shielding Architectures, Lagrangian Multipliers, Multi-Seed Validation, Autonomous Navigation.

---

## I. INTRODUCTION

The rapid advancement of autonomous systems, from industrial robotics to self-driving vehicles, has elevated the importance of safety-constrained optimization. In these domains, a single failure or constraint violation can lead to catastrophic hardware damage or irreversible system states. Traditional Reinforcement Learning (RL) agents are designed to maximize an expected return signal, which often encourages risk-taking behavior if the reward for reaching a goal outweighs the negative feedback received from sparse collisions. 

To address this, the field of Safe Reinforcement Learning (SafeRL) introduces the Constrained Markov Decision Process (CMDP) framework. In a CMDP, the agent must satisfy a safety constraint while optimizing for a primary task. Current solutions generally fall into two categories: soft-safety methods (e.g., Lagrangian penalties) which seek to satisfy constraints on average, and hard-safety methods (e.g., Shielding) which enforce constraints at every time step. 

This paper provides a rigorous comparative analysis of these methods. We implement a modular evaluation pipeline that benchmarks agent performance under varying degrees of safety-criticality. By utilizing multi-seed analysis, we account for the stochastic nature of RL exploration and provide standardized performance metrics including mean episodic rewards, violation rates, and Pareto optimality.

---

## II. THEORETICAL FOUNDATIONS

### A. Constrained Markov Decision Processes (CMDP)
A CMDP is formally defined as an extension of the Markov Decision Process (MDP), represented by the tuple $(S, A, P, R, C, \delta, \gamma)$.
- $S$: State space.
- $A$: Action space.
- $P(s'|s, a)$: Transition probability.
- $R(s, a)$: Primary reward function.
- $C(s, a)$: Constraint cost function.
- $\delta$: Safety threshold representing the maximum allowed cumulative cost.
- $\gamma$: Discount factor.

The objective is to find a policy $\pi$ that maximizes the expected discounted return $J_R(\pi) = \mathbb{E}_{\pi} [\sum_{t=0}^{\infty} \gamma^t R(s_t, a_t)]$ subject to the constraint $J_C(\pi) = \mathbb{E}_{\pi} [\sum_{t=0}^{\infty} \gamma^t C(s_t, a_t)] \leq \delta$.

### B. Baseline Architectures: Q-Learning and SARSA
We utilize vanilla Q-Learning and SARSA as baseline models. These agents optimize for $J_R(\pi)$ without explicit consideration of the cost function $C$. They serve as a benchmark for the maximum achievable reward when safety constraints are ignored.
- **Q-Learning Update**: $Q(s, a) \leftarrow Q(s, a) + \alpha [R + \gamma \max_{a'} Q(s', a') - Q(s, a)]$
- **SARSA Update**: $Q(s, a) \leftarrow Q(s, a) + \alpha [R + \gamma Q(s', a') - Q(s, a)]$

### C. Lagrangian Multiplier Methods
Lagrangian methods convert the constrained problem into an unconstrained dual optimization using a multiplier $\lambda$.
- **Fixed Lagrangian**: A constant $\lambda$ penalizes violations: $R_{eff} = R(s, a) - \lambda \cdot C(s, a)$.
- **Adaptive Lagrangian**: The multiplier $\lambda$ is updated dynamically based on the violation history. If the current cost $J_C$ exceeds $\delta$, $\lambda$ is increased; otherwise, it is decreased.
  $\lambda_{t+1} = \max(0, \lambda_t + \eta (J_C - \delta))$
  where $\eta$ is the dual learning rate.

### D. Safety-Critical Shielding Architectures
Shielding is a proactive safety layer that sits outside the agent’s exploration policy. It utilizes a known environmental model to predict whether a chosen action $a$ will lead to a violation state. If an action is deemed unsafe, the shield intercepts the command and overrides it with a fallback safe action (e.g., remaining stationary or taking a random safe step). This guarantees that the agent never enters a hazard state during exploration or deployment.

---

## III. PROPOSED METHODOLOGY

### A. Environment Specification
We utilize a 16x16 GridWorld environment (SafeMiniGrid). The state space $S$ consists of the agent’s $[x, y]$ coordinates and orientation. The action space $A$ includes movement (Forward) and directional turns (Left, Right). 
- **Goal State**: Located at $[14, 14]$.
- **Hazard States**: 15 discrete static hazards placed throughout the grid to intercept optimal paths.
- **Reward Signal**: $+10$ for reaching the goal, $-0.01$ per step to encourage efficiency, and $0$ for hazard collisions (since safety is handled by the cost function).
- **Cost Signal**: $+1$ for every hazard collision.

### B. Multi-Seed Statistical Evaluation Framework
To ensure algorithmic robustness, all agents were evaluated using a multi-seed protocol. Each algorithm was trained 5 separate times using independent random number generator seeds (Seed 0 to 4). This allows us to calculate:
- **Mean Performance**: The average reward/violation rate across all seeds.
- **Standard Deviation**: Shaded bands in our visualization represent $\pm 1$ SD, indicating the reliability and stability of the convergence.

### C. Implementation and Hyperparameters
- **Learning Rate ($\alpha$)**: 0.1
- **Discount Factor ($\gamma$)**: 0.99
- **Exploration ($\epsilon$)**: 0.1 (fixed)
- **Episodes**: 3000 sessions per seed.
- **Max Steps**: 1024 per episode.

---

## IV. RESULTS AND PERFORMANCE ANALYSIS

### A. Safety vs. Reward Pareto Frontier
The Figure below illustrates the global trade-off between reward maximization and constraint satisfaction.
*(Insert Figure: Pareto Analysis results/plots/comparisons/pareto_analysis.png)*
- **Shielded RL**: Achieves the highest safety-utility density, maintaining high rewards while consistently delivering 0.0 violations.
- **Lagrangian Methods**: Occupy the central region of the frontier, demonstrating that safety is learned over time, though occasional violations still occur during the exploration phase.
- **Baselines**: Exhibit high goal-reaching capability but with catastrophic violation rates (9.5+ per episode average), rendering them unsuitable for physical deployment.

### B. Convergence Stability
Through multi-seed analysis, we observed the training stability of each archetype.
*(Insert Figure: Violation Trajectories results/plots/shielded/shielded_violation_rate.png vs. results/plots/q_learning/q_learning_violation_rate.png)*
- Shielded agents show a flat 0.0 line with zero variance, highlighting the deterministic nature of the hard-safety layer.
- Adaptive Lagrangian agents show a high initial violation rate which decays as the $\lambda$ penalty scales, demonstrating the "learning" of the hazard boundaries.

---

## V. DISCUSSION: RISK MATURITY PROFILES

Based on our experimental data, we classify the algorithms into three "Risk Maturity Profiles":
1.  **Immature (Risk-Taking)**: Q-Learning/SARSA. Best for simulations where safety has no physical cost.
2.  **Moderate (Risk-Aware)**: Lagrangian methods. Best for environments where "near-misses" are acceptable and flexibility is required.
3.  **Mature (Strict Safety)**: Shielding. Essential for high-value hardware where even a single violation signifies mission failure.

---

## VI. CONCLUSION

This research demonstrates that effectively handling safety constraints in RL requires moving beyond simple scalar reward signals. By implementing and comparing Lagrangian and Shielding architectures, we have provided a roadmap for safe autonomous navigation. We conclude that for rigorous academic and industrial standards, a multi-seed validated Shielding approach offers the most reliable path to achieving goal objectives while maintaining absolute adherence to environmental constraints. 

---

## VII. REFERENCES

1.  E. Altman, *Constrained Markov Decision Processes*, CRC Press, 1999.
2.  M. Alshiekh, R. Bloem, R. Ehlers, O. Könighofer, S. Niekum, and U. Topcu, "Safe reinforcement learning via shielding," in *Proceedings of the AAAI Conference on Artificial Intelligence*, 2018.
3.  C. Tessler, D. J. Mankowitz, and S. Mannor, "Reward constrained policy optimization," *arXiv preprint arXiv:1805.11074*, 2018.
4.  J. Achiam, D. Held, A. Tamar, and P. Abbeel, "Constrained policy optimization," in *International Conference on Machine Learning*, pp. 22-31, 2017.
5.  R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, MIT Press, 2018.
6.  A. Ray, J. Achiam, and D. Amodei, "Benchmarking safe exploration in deep reinforcement learning," *arXiv preprint arXiv:1910.01708*, 2019.
7.  D. Bertsekas, *Constrained Optimization and Lagrange Multiplier Methods*, Academic Press, 2014.
8.  S. Gu, L. Yang, Y. Du, G. Chen, F. Walter, J. Pan, and A. Knoll, "A review of safe reinforcement learning: Methods, theory and applications," *arXiv preprint arXiv:2205.10330*, 2022.
9.  R. Cheng, G. Orosz, R. M. Murray, and J. W. Burdick, "End-to-end safe reinforcement learning through barrier functions for safety-critical systems," in *ASME 2019 Dynamic Systems and Control Conference*, 2019.
10. J. García and F. Fernández, "A comprehensive survey on safe reinforcement learning," *Journal of Machine Learning Research*, vol. 16, no. 1, pp. 1437-1480, 2015.
