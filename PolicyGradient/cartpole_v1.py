import time

import gymnasium as gym
import numpy as np
import torch
from a3c import GlobalNet

a = np.array([1, 2, 3])
# Create the CartPole environment with render_mode="human"
env = gym.make("CartPole-v1", render_mode="human")

# Define the model again (should match saved model architecture)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
policy_net = GlobalNet(state_dim, action_dim)  # Create a new instance

# Load the saved weights into this model
policy_net.load_state_dict(torch.load("PolicyGradient/a3c.pth"))

# Set the model to evaluation mode
policy_net.eval()

# Simulate the environment
for episode in range(10):  # Run for 3 episodes
    state, info = env.reset()  # Reset the environment
    done = False
    total_reward = 0
    step = 0

    print(f"\nEpisode {episode + 1}:")
    while not done:
        if isinstance(state, tuple):
            state = state[0]
        state = torch.tensor(state).float()

        env.render()  # Render the environment visually

        # Take a random action
        action, _ = policy_net(state)
        action = torch.multinomial(action, 1).item()

        # Perform the action
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated  # Combine termination and truncation

        total_reward += reward
        step += 1

        state = next_state

        print(f"Step {step}: Action={action}, Reward={reward}, Done={done}")

        if done:
            print(f"Episode finished after {step} steps. Total Reward: {total_reward}")
            break

    time.sleep(1)  # Pause between episodes

env.close()
print("\nSimulation Complete!")
