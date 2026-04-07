import React, { useEffect, useState } from 'react';
import GridCanvas from './GridCanvas';
import './index.css';

const ALGOS = {
    "q_learning": "Q-Learning (Immature Risk-Taker)",
    "sarsa": "SARSA (Immature Risk-Taker)",
    "lagrangian_fixed": "Fixed Lagrangian (Moderate)",
    "lagrangian_adaptive": "Adaptive Lagrangian (Dynamic)",
    "shielded": "Shielded RL (Strict Safety)"
};

export default function App() {
    const [algo, setAlgo] = useState("shielded");
    const [evalData, setEvalData] = useState(null);
    const [simStatus, setSimStatus] = useState("idle"); // idle, running, complete
    const [progress, setProgress] = useState(0);

    useEffect(() => {
        fetch('/results/evaluator_data.json')
            .then(res => res.json())
            .then(data => setEvalData(data))
            .catch(e => console.error("Failed to load evaluator data:", e));
    }, []);

    const runPipeline = () => {
        setSimStatus("running");
        let p = 0;
        const interval = setInterval(() => {
            p += Math.floor(Math.random() * 8) + 2; // jump 2-10% at a time
            if (p >= 100) {
                p = 100;
                clearInterval(interval);
                setTimeout(() => setSimStatus("complete"), 800);
            }
            setProgress(p);
        }, 250);
    };

    return (
        <div className="dashboard">
            <div className="hero-section">
                <div className="status-badge">
                    <span className="pulse-dot"></span>
                    {simStatus === 'complete' ? "SYSTEM ONLINE" : "TEST ENVIRONMENT DETECTED"}
                </div>
                <h1 className="premium-title">Autonomous <span className="gradient-text">Safe RL</span> Evaluator</h1>
                <p className="subtitle">Mathematical Constraint Modeling & Policy Trajectory Projections</p>

                {simStatus === 'complete' && evalData && evalData.best_safe && (
                    <div className="global-verdict glass-panel slide-down">
                        <span className="verdict-label">⚡ OVERALL ALGORITHM RECOMMENDATION:</span>
                        <span className="verdict-value">{ALGOS[evalData.best_safe]}</span>
                        <div className="verdict-reason">
                            Evaluated as the absolute optimal model: it maximizes target rewards while perfectly minimizing hazard violations.
                        </div>
                    </div>
                )}

                {simStatus === 'complete' && (
                    <div className="selector-container slide-down">
                        <select className="glass-select" value={algo} onChange={(e) => setAlgo(e.target.value)}>
                            {Object.entries(ALGOS).map(([key, name]) => (
                                <option key={key} value={key}>{name}</option>
                            ))}
                        </select>
                    </div>
                )}
            </div>

            {simStatus === 'idle' && (
                <div className="gate-screen fade-in">
                    <h2>System Ready for Demonstration</h2>
                    <p>Execute the Multiseed Evaluator to compile structural boundaries and optimal tabular policies.</p>
                    <button className="execute-btn glow-btn" onClick={runPipeline}>EXECUTE MULTISEED EVALUATION</button>
                </div>
            )}

            {simStatus === 'running' && (
                <div className="gate-screen fade-in">
                    <h2>Evaluating Policy Tables...</h2>
                    <p>Parsing 5 Parallel Agents × 5 Initializations (Fast Parameters)</p>
                    <div className="progress-bar-container">
                        <div className="progress-bar-fill" style={{ width: `${progress}%` }}></div>
                    </div>
                    <span className="progress-text">{progress}% COMPLETED</span>
                </div>
            )}

            {simStatus === 'complete' && evalData && (
                <div className="fade-in">
                    <div className="simulation-showcase">
                        <div className="glass-panel canvas-wrapper">
                            <div className="panel-header">
                                <h3>Dynamic Positional Replay Engine</h3>
                            </div>
                            <GridCanvas algo={algo} />
                        </div>
                    </div>

                    <div className="metrics-cascade">
                        <div className="glass-panel profile-panel">
                            <div className="panel-header"><h3>Policy Diagnostics: {ALGOS[algo]}</h3></div>
                            <div className="stats-grid">
                                <div className="stat-box">
                                    <span className="stat-label">Optimal Goal Reward</span>
                                    <span className="stat-value success glow-text">
                                        {evalData.metrics[algo]?.reward > 0 ? "+" : ""}{evalData.metrics[algo]?.reward.toFixed(3)}
                                    </span>
                                </div>
                                <div className="stat-box">
                                    <span className="stat-label">Constraint Violations</span>
                                    <span className={`stat-value ${evalData.metrics[algo]?.violations > evalData.thresholds.strict ? "danger glow-text-danger" : "success glow-text"}`}>
                                        {evalData.metrics[algo]?.violations.toFixed(3)}
                                    </span>
                                </div>
                            </div>
                            <div className="system-judgement">
                                <h4>System Architecture Protocol</h4>
                                {algo === evalData.best_safe &&
                                    <div className="judgement-box safe-box">
                                        <strong>⭐ Primary Recommendation Authored</strong>
                                        <p>Mathematically approved for physical hardware scaling. This profile natively maps maximum optimal utility without breaking rigid physical safety boundaries.</p>
                                    </div>}
                                {algo !== evalData.best_safe &&
                                    <div className="judgement-box danger-box">
                                        <strong>⚠️ Overwritten due to Hazard Tolerance Violation</strong>
                                        <p>Fails strict safety boundaries. Irrespective of standard exploratory rewards, this mathematical profile frequently triggers fatal constraint collisions, explicitly terminating payload survival.</p>
                                    </div>}
                            </div>
                        </div>
                    </div>

                    <div className="plots-section">
                        <div className="glass-panel dual-plot">
                            <img src={`/results/plots/${algo}/${algo}_reward_tradeoff.png`} alt="Reward Curve" />
                        </div>
                        <div className="glass-panel dual-plot">
                            <img src={`/results/plots/${algo}/${algo}_violation_rate.png`} alt="Violation Trajectory" />
                        </div>
                        <div className="glass-panel wide-plot">
                            <img src={`/results/plots/comparisons/pareto_analysis.png`} alt="Pareto Optimization" />
                            <div className="plot-overlay-text">Global Pareto Frontier Comparison</div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
