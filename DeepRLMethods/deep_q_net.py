import random
from collections import deque

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        """
        Implementation of a Deep Q-Network (DQN) for Q-Learning.
        Args:
            state_dim: Dimension of the state space.
            action_dim: Dimension of the action space.
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)  # Output Q-values for all actions

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Using ReLU activation function.
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # No activation (raw Q-values)


class ReplayBuffer:
    def __init__(self, size):
        """
        Replay buffer to store and sample experience tuples.
        Args:
            size: Maximum number of experience tuples to store.
        """
        self.size = size
        self.buffer = deque(maxlen=size)

    def push(self, state, action, reward, next_state, done):
        """
        Add a new experience tuple to the buffer.
        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received after taking the action.
            next_state: Next state.
            done: Whether the episode has terminated.
        """
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size) -> tuple:
        """
        Sample a batch of experience tuples from the buffer.
        Args:
            batch_size: Number of experience tuples to sample.
        Returns:
            A batch of experience tuples.
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)

        # Convert each component into tensors
        state = torch.tensor(np.array(state), dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.int64)
        reward = torch.tensor(reward, dtype=torch.float32)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)

        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


def train_dqn(
    env,
    num_episodes=1500,
    batch_size=64,
    gamma=0.9,
    lr=0.0001,
    epsilon_decay=0.995,
    min_epsilon=0.01,
):
    """
    Function to train a DQN agent on a given environment.
    Args:
        env: Gym environment.
        num_episodes: Number of episodes to train the agent.
        batch_size: Number of experience tuples to sample from the replay buffer.
        gamma: Discount factor.
        lr: Learning rate for the optimizer.
        epsilon_decay: Decay rate for epsilon.
        min_epsilon: Minimum value for epsilon.
    Returns:
        Trained DQN agent, list of episode rewards.
    """
    rewards = []  # Rewards per episode
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Initialize DQN and optimizer
    q_network = DQN(state_dim, action_dim)
    target_network = DQN(state_dim, action_dim)
    target_network.load_state_dict(q_network.state_dict())

    optimizer = torch.optim.Adam(q_network.parameters(), lr=lr)
    criterion = nn.MSELoss()
    replay_buffer = ReplayBuffer(size=1000)

    epsilon = 1.0  # Start with full exploration
    # Update target network every 10 episodes (Can be done based on steps also)
    target_update_freq = 10

    for episode in range(num_episodes):
        state = env.reset()  # Reset the env to original state before every episode.
        if isinstance(state, tuple):
            state = state[0]

        done = False  # Episode termination flag
        episode_reward = 0  # Reward accumulated in the episode.
        episode_loss = 0

        while not done:
            # Epsilon-greedy policy
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                # Get the action with max Q-value for the current state.
                action = torch.argmax(q_network(torch.tensor(state).float())).item()

            next_state, reward, done, _, _ = env.step(action)
            episode_reward += reward

            if isinstance(next_state, tuple):
                next_state = next_state[0]

            # Store the experience in the replay buffer
            replay_buffer.push(state, action, reward, next_state, done)

            # Update the network weights if replay buffer has enough samples.
            if len(replay_buffer) > batch_size:
                (
                    state_batch,
                    action_batch,
                    reward_batch,
                    next_state_batch,
                    done_batch,
                ) = replay_buffer.sample(batch_size)

                # Compute Q-values for the current state and next state
                q_values = q_network(state_batch).gather(1, action_batch.unsqueeze(1))
                # Compute target Q-values using target network
                with torch.no_grad():
                    max_next_q_values = target_network(next_state_batch).max(1)[0]
                    target_q_values = reward_batch + gamma * max_next_q_values * (
                        1 - done_batch.float()
                    )

                # Compute loss
                loss = criterion(q_values, target_q_values.unsqueeze(1))
                episode_loss += loss.item()

                # Update network
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            state = next_state  # Move to the next state

        # Decay exploration rate
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        # Update target network every few episodes
        if episode % target_update_freq == 0:
            target_network.load_state_dict(q_network.state_dict())

        rewards.append(episode_reward)
        if episode % 10 == 0:
            print(
                f"Episode {episode}, Reward: {episode_reward}, Loss: {episode_loss:.3f}"
            )

    return q_network, rewards


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    trained_q_network, rewards = train_dqn(env)
    torch.save(trained_q_network.state_dict(), "DeepRLMethods/dqn_cartpole.pth")

    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN Training Performance on CartPole")
    plt.show()
