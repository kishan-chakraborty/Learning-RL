"""
10 ARM BANDIT TEST BED
SOURCE:
    Reinforcement learning: An introduction by Sutton and Barto. 
"""
import numpy as np

class Bandit:
    """
    Defining the multi-arm bandit (slot machine) class.
    """
    def __init__(self, n_arms:int=10, mean:float=0, std_dev:float=1)->None:
        """
        Args:
            n_arms: Number of arms
            mean: Mean of the normal distribution generating true action values.
            std: std of the normal distribution generating true action values.
            
        Returns:
            None
        """
        self.n_arms = n_arms
        self.true_action_values = np.random.normal(mean, std_dev, n_arms)
