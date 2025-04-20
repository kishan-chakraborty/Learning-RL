import gymnasium as gym
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torch.optim import Adam


class GlobalNet(nn.Module):
    def __init__(self, state_dim: int, n_actions: int):
        """
        Implement the glocal actor network
        Args:
            input_size: dimension of state.
            action_size: No. of actions.
        """
        super(GlobalNet, self).__init__()
        # Implementing commong network
        self.common = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU()
        )
        self.actor = nn.Linear(128, n_actions)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = self.common(x)
        action_logits = torch.softmax(self.actor(x), dim=-1)
        val = self.critic(x)

        return action_logits, val


def calculate_rewards(rewards: list, gamma: float, final_val: float) -> list:
    """
    Calculate the total discounted rewards for eact time steps

    Args:
        rewards: List of rewards received during the episode.
        gamma: Discount factor.

    Returns:
        list of discounted rewards.
    """
    total_rewards = []
    running_total = final_val
    for r in reversed(rewards):
        running_total = r * gamma * running_total
        total_rewards.insert(0, running_total)

    return total_rewards


def train_model(
    env, global_net, worker_id: int, n_epiisodes: int, update_global: int, gamma: float
):
    """
    Implement the training process for each worker
    Args:
        env: gym environment.
        global_net: global network.
        worker_id: worker id.
        n_episoeds: No. of episodes per user.
        update_global: No. of steps after which the global network is updated.
        gamma: Discount factor.
    """
    # Create the local network
    local_net = GlobalNet(env.observation_space.shape[0], env.action_space.n)
    local_net.load_state_dict(global_net.state_dict())

    optimizer1 = Adam(local_net.parameters(), lr=1e-3)
    optimizer2 = Adam(local_net.parameters(), lr=1e-3)

    for episode in range(n_epiisodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        log_probs, rewards, values = [], [], []
        done = False
        step_count = 0

        while not done:
            step_count += 1
            state = torch.tensor(state, dtype=torch.float32)
            action_logits, value = local_net(state)
            action = torch.multinomial(action_logits, 1).item()

            next_state, reward, done, _, _ = env.step(action)
            if isinstance(next_state, tuple):
                next_state = next_state[0]

            log_probs.append(torch.log(action_logits[action]))
            values.append(value)
            rewards.append(reward)

            state = next_state

            if done or step_count % update_global == 0:
                # Calculate the discounted rewards
                if done:
                    discounted_rewards = calculate_rewards(rewards, gamma, 0)
                else:
                    _, next_value = local_net(
                        torch.tensor(next_state, dtype=torch.float32)
                    )
                    discounted_rewards = calculate_rewards(rewards, gamma, next_value)

                # Calculate the advantage
                value_tensor = torch.stack(values).squeeze(-1)
                advantages = (
                    torch.tensor(discounted_rewards, dtype=torch.float32) - value_tensor
                )

                # Calculate actor loss
                entropy = -torch.sum(action_logits * torch.log(action_logits))
                actor_loss = (
                    -torch.sum(torch.stack(log_probs) * advantages.detach())
                    + 0.001 * entropy
                )
                optimizer1.zero_grad()
                actor_loss.backward()
                optimizer1.step()

                # Calculate critic loss
                critic_loss = torch.sum(advantages.detach() ** 2)
                critic_loss.requires_grad = True
                optimizer2.zero_grad()
                critic_loss.backward()
                optimizer2.step()

                # Update the global network
                global_net.load_state_dict(local_net.state_dict())

                # Clear the lists
                log_probs, rewards, values = [], [], []

            if worker_id == 0 and episode % 10 == 0:
                print(f"n_episodes: {episode}, Rewards: {sum(rewards)}")


if __name__ == "__main__":
    mp.set_start_method("spawn")  # Required for multiprocessing

    # Defie the environment and variables
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Create the global network
    global_net = GlobalNet(state_dim, n_actions)
    global_net.share_memory()

    n_workers = mp.cpu_count()

    processes = []
    for i in range(n_workers):
        p = mp.Process(
            target=train_model,
            args=(env, global_net, i, 500, 100, 0.99),
        )
        p.start()
        processes.append(p)

    # Wait for all workers to finish
    for p in processes:
        p.join()

    print("Training Complete!")

    torch.save(
        global_net.state_dict(),
        "PolicyGradient/a3c.pth",
    )
