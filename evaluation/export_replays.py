import pickle
import json
import numpy as np
import os
from env.safe_minigrid import SafeMiniGridEnv
from agents.q_learning import get_state
from agents.shielding import select_shielded_action

ALGORITHMS = ["q_learning", "sarsa", "lagrangian_fixed", "lagrangian_adaptive", "shielded"]

def export_replays():
    os.makedirs("results/replays", exist_ok=True)
    env = SafeMiniGridEnv(size=16)

    for algo in ALGORITHMS:
        q_path = f"results/{algo}_Q.pkl"
        if algo == "shielded":
            q_path = "results/shielded_q_learning_Q.pkl"
            
        if not os.path.exists(q_path):
            print(f"Skipping {algo} replay export, no Q table found.")
            continue
            
        with open(q_path, "rb") as f:
            Q = pickle.load(f)
            
        env.reset(seed=42)
        state = get_state(env)
        
        history = []
        done = False
        steps = 0
        
        # Initial pos
        history.append({
            "x": int(env.agent_pos[0]), 
            "y": int(env.agent_pos[1]), 
            "dir": int(env.agent_dir),
            "action": -1,
            "reward": 0,
            "violation": 0
        })

        while not done and steps < 150:
            if algo == "shielded":
                action, _ = select_shielded_action(Q, state, 0.0, env)
            else:
                # Default defaultdict behavior for unfound states (zeros)
                vals = Q.get(state, np.zeros(env.action_space.n))
                action = int(np.argmax(vals))
                
            _, reward, terminated, truncated, info = env.step(action)
            state = get_state(env)
            
            history.append({
                "x": int(env.agent_pos[0]), 
                "y": int(env.agent_pos[1]), 
                "dir": int(env.agent_dir),
                "action": int(action),
                "reward": float(reward),
                "violation": int(info["violation"])
            })
            
            steps += 1
            done = terminated or truncated
            
        with open(f"results/replays/{algo}.json", "w") as f:
            json.dump(history, f, indent=2)
            
        print(f"Exported {steps} steps for {algo}")

if __name__ == "__main__":
    export_replays()
