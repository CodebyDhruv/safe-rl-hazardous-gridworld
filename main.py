from env.safe_minigrid import SafeMiniGridEnv

def main():
    env = SafeMiniGridEnv(render_mode="human")
    obs, _ = env.reset(seed=42)

    done = False
    while not done:
        action = env.action_space.sample() 
        obs, reward, terminated, truncated, info = env.step(action)

        print(
            f"Pos: {info['agent_pos']} | "
            f"Reward: {reward:.2f} | "
            f"Violation: {info['violation']}"
        )

        done = terminated or truncated

    input("Episode finished. Press Enter to close.")
    env.close()

if __name__ == "__main__":
    main()
