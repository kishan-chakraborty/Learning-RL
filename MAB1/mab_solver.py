"""
Solver for multiarm bandit problem. Using the following methods:
Greedy
Epsilon Greedy
UCB
Optimistic Initialization.

Source:
    Reinforcement learning: An introduction by Sutton and Barto.
"""

import numpy as np
import matplotlib.pyplot as plt
from bandit import Bandit

class MABSolver:
    """
    MULTI-ARM BANDIT SOLVER
    """
    def __init__(self,
                 n_arms:int=10,
                 n_experiments:int=2000,
                 n_steps:int=1000)->None:
        """
        Args:
            n_arms: No. of bandit arms.
            n_experiments: Number of bandit instances.
            n_steps: Number of time steps per bandit instance.
        Returns:
            None
        """
        self.n_arms = n_arms
        self.n_experiments = n_experiments
        self.n_steps = n_steps

        # Optimistic initialization of reward estimates.
        self.initial_estimation = None

        # Current bandit instance.
        self.bandit = None

        # Average reward over the experiments for each time step.
        self.total_rewards = None

        # Rewards for eact time step.
        self.rewards = None

        # Count of each action for each time step.
        self.action_counts = None

        # initial estimated values.
        self.estimated_values = None


    def _initialize(self, initial_estimation: list = None)->None:
        """
        Initializing required variables for the solver.
        """
        # Assert initial_estimates is None or equal to no. of bandit arms
        assert initial_estimation is None or len(initial_estimation) == self.n_arms

        # If no initial estimates are provided, initialize all to 0.
        if initial_estimation is None:
            self.initial_estimation = np.zeros(self.n_arms)
        else:
            self.initial_estimation = np.array(initial_estimation, dtype=float)

        # Average reward over the experiments for each time step.
        self.total_rewards = np.zeros(self.n_steps)

        # Rewards for eact time step.
        self.rewards = np.zeros(self.n_steps)

        # Count of each action for each time step.
        self.action_counts = np.zeros(self.n_arms)

        # initial estimated values.
        self.estimated_values = np.copy(self.initial_estimation)


    def calculate_reward(self,
                        epsilon:float=None,
                        c:float=None,
                        step:int=None)->tuple:
        """
        Calculate reward for the next time step.
        Args:
            epsilon: Exploration rate [In case of epsilon-greedy].
            c: Exploration parameter [In case of UCB].
        """
        # UCB action selection
        if c is not None:
            # Make sure each arm (action) is slotted at least once.
            total_action = int(self.action_counts.sum())

            if total_action < self.n_arms:
                action = total_action
            else:
                # Select the action with highest average reward
                action = np.argmax(self.estimated_values + c * np.sqrt(np.log(step) \
                                                            / (self.action_counts)))
        # Epsilon-greedy method action selection
        elif epsilon is not None and np.random.rand() < epsilon:
            action = int(np.random.choice(self.n_arms))
        else:
            action = np.argmax(self.estimated_values)

        # reward + gaussian noise.
        reward = self.bandit.true_action_values[action] + np.random.normal(0, 1)

        # Count of current action
        self.action_counts[action] += 1
        return reward, action


    def greedy(self,
               epsilon: float = None,
               initial_estimation: list = None,
               alpha:float=None)->list:
        """
        Solving the multi-armed bandit problem using the greedy method.
        Exploitation no exploration unless optimistic initialization is used.

        Args:
            epsilon: Exploration rate.
            initial_estimation: Optimistic initial estimates.
            alpha: step size in case of Fixed step size. 
                    If None then incremental implementation is used.

        Returns:
            list of rewards for each time step averaging over the number of experiments.
        """
        total_rewards = np.zeros(self.n_steps)

        # Running experiments. The final result is the average of all experiments.
        for _ in range(self.n_experiments):
            # Initialize the bandit for the current experiment.
            self.bandit = Bandit(self.n_arms)
            self._initialize(initial_estimation)

            for step in range(self.n_steps):
                if epsilon is None:
                    reward, action = self.calculate_reward()
                else:
                    reward, action = self.calculate_reward(epsilon=epsilon)

                self.rewards[step] = reward  # Storing reward for each time step

                if alpha is None:
                    # Incremental implementation
                    self.estimated_values[action] += (reward - self.estimated_values[action])\
                                                     /self.action_counts[action]
                else:
                    # Fixed step implementation.
                    self.estimated_values[action] += (reward - self.estimated_values[action]) \
                                                    * alpha

            # Sum rewards over all experiments
            total_rewards += self.rewards
        avg_rewards = total_rewards / self.n_experiments
        return avg_rewards


    def ucb(self,
            initial_estimation: list = None,
            c: float=2.0)->list:
        """
        Solving the multi-armed bandit problem using the UCB method.

        Args:
            initial_estimation: Optimistic initial estimates.
            c: Exploration parameter.

        Returns:
            list of rewards for each time step averaging over the number of experiments.
        """
        total_rewards = np.zeros(self.n_steps)

        # Running experiments. The final result is the average of all experiments.
        for _ in range(self.n_experiments):
            # Initialize the bandit for the current experiment.
            self.bandit = Bandit(self.n_arms)
            self._initialize(initial_estimation)

            for step in range(self.n_steps):
                reward, action = self.calculate_reward(c=c, step=step)
                self.rewards[step] = reward  # Storing reward for each time step
                # Incremental implementation
                self.estimated_values[action] += (reward - self.estimated_values[action]) \
                                                / self.action_counts[action]

            # Sum rewards over all experiments
            total_rewards += self.rewards
        avg_rewards = total_rewards / self.n_experiments

        return avg_rewards


if __name__ == '__main__':
    solver = MABSolver()
    ucb_sol = solver.ucb()
    # eps_greedy_sol = solver.greedy(epsilon=0.1)
    plt.plot(ucb_sol[:15])
    # plt.plot(eps_greedy_sol)
    plt.show()
