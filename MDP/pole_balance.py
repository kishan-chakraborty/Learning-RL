import time

import gymnasium as gym
import numpy as np

a = np.array([1, 2, 3])
# Create the CartPole environment with render_mode="human"
env = gym.make("CartPole-v1", render_mode="human")

# Simulate the environment
for episode in range(3):  # Run for 3 episodes
    state, info = env.reset()  # Reset the environment
    done = False
    total_reward = 0
    step = 0

    print(f"\nEpisode {episode + 1}:")
    while not done:
        env.render()  # Render the environment visually

        # Take a random action
        action = env.action_space.sample()

        # Perform the action
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated  # Combine termination and truncation

        total_reward += reward
        step += 1

        print(f"Step {step}: Action={action}, Reward={reward}, Done={done}")

        if done:
            print(f"Episode finished after {step} steps. Total Reward: {total_reward}")
            break

    time.sleep(1)  # Pause between episodes

env.close()
print("\nSimulation Complete!")
