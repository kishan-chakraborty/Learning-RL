import time

import gymnasium as gym
import torch
from ddpg import ActorNet, train_ddpg


def train_network(env):
    # Train DDPG for the mountain car environment
    trained_network, _ = train_ddpg(env)

    torch.save(trained_network.state_dict(), "DPG/ddpg/mountain_car.pth")


# Create the mountain car environment with render_mode="human"
env = gym.make("MountainCarContinuous-v0")

# train_network(env)

env = gym.make("MountainCarContinuous-v0", render_mode="human")

# Define the model again (should match saved model architecture)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
policy_net = ActorNet(state_dim, action_dim)  # Create a new instance

# Load the saved weights into this model
policy_net.load_state_dict(torch.load("DPG/ddpg/mountain_car.pth"))

# Set the model to evaluation mode
policy_net.eval()

# Simulate the environment
for episode in range(10):  # Run for 3 episodes
    state, info = env.reset()  # Reset the environment
    done = False
    total_reward = 0
    step = 0
    max_action = env.action_space.high[0]

    print(f"\nEpisode {episode + 1}:")
    while not done:
        env.render()  # Render the environment visually

        # Take a random action
        action = max_action * policy_net(torch.tensor(state)).detach().numpy()

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
