import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        """
        Implementation of a policy network for policy gradient methods.
        Args:
            state_dim: Dimension of the state space.
            action_dim: Dimension of the action space.
        """
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=-1)
        return x


def calculate_rewards(rewards, gamma):
    """
    Calculate the total discounted rewards for each time step.
    Args:
        rewards: List of rewards received during the episode.
        gamma: Discount factor.
    Returns:
        List of total discounted rewards.
    """
    total_rewards = []
    running_total = 0
    for r in reversed(rewards):
        running_total = r + gamma * running_total
        total_rewards.insert(0, running_total)
    return total_rewards


def policy_gradient(env, alpha, gamma, num_episodes):
    """
    Implementation of the REINFORCE algorithm for policy gradient methods.
    Args:
        env: OpenAI Gym environment.
        alpha: Learning rate.
        gamma: Discount factor.
        num_episodes: Number of episodes to train the agent.
    Returns:
        The trained policy network.
    """
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    policy_net = PolicyNet(state_dim, action_dim)
    optimizer = optim.Adam(policy_net.parameters(), lr=alpha)
    reward_history = []

    for episode in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        done = False
        rewards = []  # store reward values to calculate total discounted rewards.
        log_probs = []

        while not done:
            state = torch.tensor(np.array(state), dtype=torch.float32)
            action_prob = policy_net(state)
            action = torch.multinomial(action_prob, 1).item()

            next_state, r, done, _, _ = env.step(action)
            if isinstance(next_state, tuple):
                next_state = next_state[0]

            rewards.append(r)
            log_probs.append(torch.log(action_prob[action]))

            state = next_state

        # Calculate the total discounted rewards
        total_rewards = calculate_rewards(rewards, gamma)

        reward_history.append(total_rewards[0])

        # Update the policy network
        optimizer.zero_grad()
        loss = -torch.sum(
            torch.stack(log_probs) * torch.tensor(np.array(total_rewards))
        )
        loss.backward()
        optimizer.step()

        # Print the episode reward
        if episode % 10 == 0:
            print(f"Episode: {episode}, Reward: {sum(rewards)}")

    return policy_net, reward_history


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    alpha = 0.0001
    gamma = 0.99
    num_episodes = 2000

    trained_policy_net, rewards = policy_gradient(env, alpha, gamma, num_episodes)

    torch.save(trained_policy_net.state_dict(), "PolicyGradient/reinforce_cartpole.pth")

    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("REINFORCE Training Performance on CartPole")
    plt.show()
