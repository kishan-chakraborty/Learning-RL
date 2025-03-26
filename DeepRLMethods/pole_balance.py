import time

import gymnasium as gym
import numpy as np
import torch
from deep_q_net import DQN

a = np.array([1, 2, 3])
# Create the CartPole environment with render_mode="human"
env = gym.make("CartPole-v1", render_mode="human")

# Define the model again (should match saved model architecture)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
policy_net = DQN(state_dim, action_dim)  # Create a new instance

# Load the saved weights into this model
policy_net.load_state_dict(
    torch.load(
        "/home/kishan/Desktop/Kishan/Projects/Reinforcement Learning/Learning-RL/DeepRLMethods/dqn_cartpole.pth"
    )
)

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
        env.render()  # Render the environment visually

        # Take a random action
        action = torch.argmax(policy_net(torch.tensor(state).float())).item()

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
