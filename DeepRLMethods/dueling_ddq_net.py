import random
from collections import deque

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class DuelingDDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        """
        Implementation of a Deep Q-Network (DQN) for Q-Learning.
        Args:
            state_dim: Dimension of the state space.
            action_dim: Dimension of the action space.
        """
        super(DuelingDDQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)

        self.value = nn.Linear(64, 1)
        self.actions = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Using ReLU activation function.
        x = torch.relu(self.fc2(x))

        value = self.value(x)
        actions = self.actions(x)
        q_values = value + actions - actions.mean(dim=1, keepdim=True)
        return q_values


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


def train_dueling_ddqn(
    env,
    num_episodes=1500,
    batch_size=64,
    gamma=0.9,
    lr=0.0001,
    epsilon_decay=0.995,
    min_epsilon=0.01,
    tau=0.01,
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
    dueling_network = DuelingDDQN(state_dim, action_dim)
    target_network = DuelingDDQN(state_dim, action_dim)
    target_network.load_state_dict(dueling_network.state_dict())

    optimizer = torch.optim.Adam(dueling_network.parameters(), lr=lr)
    criterion = nn.MSELoss()
    replay_buffer = ReplayBuffer(size=1000)

    epsilon = 1.0  # Start with full exploration

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
                action = torch.argmax(
                    dueling_network(torch.tensor([state]).float())
                ).item()

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
                q_values = dueling_network(state_batch).gather(
                    1, action_batch.unsqueeze(1)
                )
                # Compute target Q-values using target network
                with torch.no_grad():
                    max_actions = torch.argmax(
                        dueling_network(next_state_batch), axis=1
                    )
                    target_q_values = reward_batch + gamma * target_network(
                        next_state_batch
                    )[range(batch_size), max_actions] * (1 - done_batch.float())

                # Compute loss
                loss = criterion(q_values, target_q_values.unsqueeze(1))
                episode_loss += loss.item()

                # Update network
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            state = next_state  # Move to the next state

            # Update target network using polyak averaging
            for target_param, param in zip(
                target_network.parameters(), dueling_network.parameters()
            ):
                target_param.data.copy_(
                    tau * param.data + (1.0 - tau) * target_param.data
                )

        # Decay exploration rate
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        rewards.append(episode_reward)
        if episode % 10 == 0:
            print(
                f"Episode {episode}, Reward: {episode_reward}, Loss: {episode_loss:.3f}"
            )

    return dueling_network, rewards


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    trained_dueling_network, rewards = train_dueling_ddqn(env)
    torch.save(
        trained_dueling_network.state_dict(), "DeepRLMethods/dueling_ddqn_cartpole.pth"
    )

    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN Training Performance on CartPole")
    plt.show()
