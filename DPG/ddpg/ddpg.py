"""
Implementing Deep Deterministic Policy Gradient (DDPG) algorithm.
"""

import random
from collections import deque

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class ActorNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        """
        Implementation of a Actor network for DDPG.
        The output is a continuous action vector.
        Args:
            state_dim: Dimension of the state space.
            action_dim: Dimension of the action space.
        """
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 400)
        self.bn1 = nn.LayerNorm(400)
        self.fc2 = nn.Linear(400, 300)
        self.bn2 = nn.LayerNorm(300)
        self.fc3 = nn.Linear(300, action_dim)  # Output Q-values for all actions

    def forward(self, x):
        x = self.bn1(torch.relu(self.fc1(x)))  # Using ReLU activation function.
        x = self.bn2(torch.relu(self.fc2(x)))
        return torch.tanh(self.fc3(x))  # No activation (raw Q-values)


class CriticNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        """
        Implementating Critic network for DDPG.
        The output is a Q-value for the given state-action pair.
        Args:
            state_dim: Dimension of the state space.
            action_dim: Dimension of the action space.
        """
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.bn1 = nn.LayerNorm(400)
        self.fc2 = nn.Linear(400, 300)
        self.bn2 = nn.LayerNorm(300)
        self.fc3 = nn.Linear(300, 1)  # Output Q-values for all actions

    def forward(self, x):
        x = self.bn1(torch.relu(self.fc1(x)))  # Using ReLU activation function.
        x = self.bn2(torch.relu(self.fc2(x)))
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
        action = torch.tensor(action, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)

        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class OUNoise:
    def __init__(
        self, action_dim, mu=0.0, theta=0.15, sigma=0.2, sigma_min=0.05, decay=1e-4
    ):
        """
        Simplified Ornstein-Uhlenbeck noise generator.

        Args:
            action_dim (int): Number of actions.
            mu (float): Mean value (default: 0).
            theta (float): Mean reversion speed.
            sigma (float): Initial noise standard deviation.
            sigma_min (float): Minimum allowed noise.
            decay (float): Decay rate for sigma.
        """
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.sigma_min = sigma_min
        self.decay = decay
        self.state = np.ones(self.action_dim) * self.mu  # Initialize state

    def sample(self, t):
        """
        Generates OU noise based on the given time step t.

        Args:
            t (int): The current time step.

        Returns:
            np.ndarray: OU noise for the current time step.
        """
        # Decay sigma over time
        sigma_t = max(self.sigma * np.exp(-self.decay * t), self.sigma_min)

        # OU noise update equation
        dx = self.theta * (self.mu - self.state) + sigma_t * np.random.randn(
            self.action_dim
        )
        self.state += dx
        return self.state


def train_ddpg(env, num_episodes=500, batch_size=64, gamma=0.99, tau=0.001):
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
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    # Initialize the actor and critic network
    actor_net = ActorNet(state_dim, action_dim)
    critic_net = CriticNet(state_dim, action_dim)

    # Initialize the target networks
    target_actor = ActorNet(state_dim, action_dim)
    target_critic = CriticNet(state_dim, action_dim)
    target_actor.load_state_dict(actor_net.state_dict())
    target_critic.load_state_dict(critic_net.state_dict())

    # Intialize corresponding optimizers
    optimizer1 = torch.optim.Adam(actor_net.parameters(), lr=10e-4)
    optimizer2 = torch.optim.Adam(critic_net.parameters(), lr=10e-3)
    criterion = nn.MSELoss()

    # Initialize the replay buffer
    replay_buffer = ReplayBuffer(size=1000000)

    for episode in range(num_episodes):
        state = env.reset()  # Reset the env to original state before every episode.
        # For exploration use Ornstein-Uhlenbeck process.
        ou_noise = OUNoise(action_dim=action_dim)

        done, truncate = False, False  # Episode termination flag
        episode_reward = 0  # Reward accumulated in the episode.
        time_step = 0  # Time step in the episode

        while not done and not truncate:
            time_step += 1
            if isinstance(state, tuple):
                state = state[0]
            state = torch.tensor(state, dtype=torch.float32)

            # Select an action using noisy behavior policy
            action = max_action * actor_net(state)
            action_noisy = action.detach().numpy() + ou_noise.sample(time_step)

            next_state, reward, done, truncate, _ = env.step(action_noisy)
            episode_reward += reward

            # Store the experience in the replay buffer
            replay_buffer.push(state, action_noisy, reward, next_state, done)

            # Update the network weights if replay buffer has enough samples.
            if len(replay_buffer) > batch_size:
                (
                    state_batch,
                    action_batch,
                    reward_batch,
                    next_state_batch,
                    done_batch,
                ) = replay_buffer.sample(batch_size)

                # calculate the actions using the target actor network
                target_actions = target_actor(next_state_batch)

                # Calculate the Q-values using the target critic network
                q_values = target_critic(
                    torch.cat((next_state_batch, target_actions), dim=1)
                )
                target_q_values = reward_batch + (
                    1 - done_batch
                ) * gamma * q_values.squeeze(-1)

                # Calculate the Q-values using the critic network
                q_values = critic_net(torch.cat((state_batch, action_batch), dim=1))

                # Calculate actor loss
                actor_loss = -critic_net(
                    torch.cat((state_batch, max_action * actor_net(state_batch)), dim=1)
                ).mean()

                # Calculate critic loss
                critic_loss = criterion(q_values, target_q_values.unsqueeze(1))

                # Update the actor network
                optimizer1.zero_grad()
                actor_loss.backward()
                optimizer1.step()

                # Update the critic network
                optimizer2.zero_grad()
                critic_loss.backward()
                optimizer2.step()

                # Update the target networks using soft update
                for target_param, param in zip(
                    target_actor.parameters(), actor_net.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )

                for target_param, param in zip(
                    target_critic.parameters(), critic_net.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )

            state = next_state  # Move to the next state

        rewards.append(episode_reward)
        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {episode_reward}")

    return actor_net, rewards


if __name__ == "__main__":
    env = gym.make("Pendulum-v1")
    trained_q_network, rewards = train_ddpg(env)
    torch.save(trained_q_network.state_dict(), "DPG/ddpg/pendulum.pth")

    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN Training Performance on CartPole")
    plt.show()
