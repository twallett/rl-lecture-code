#%%

import numpy as np

def create_environment(**kwargs):
  
    """
    # Create synthetic data for a multi-armed bandit environment.

    Parameters:
    - env (str): The type of environment to generate ("gaussian" or "bernoulli").
    - n_arms (int): The number of arms in the bandit (default is 2).
    - arm_means (list): List of lists representing mean values for each arm in each environment.
      For "gaussian" environment, it should be a list of lists of floats.
      For "bernoulli" environment, it should be a list of lists of probabilities.
      Default is [[[0.9, 0.1],[0.1, 0.9]]].
    - random_seed (int): Seed for the random number generator (default is None).
    - observations (int): Number of observations to generate for each arm (default is 2000).

    Returns:
    - numpy.ndarray: Synthetic data matrix representing observations from the specified multi-armed bandit environment.
      For "gaussian" environment, each column corresponds to an arm.
      For "bernoulli" environment, each column corresponds to a binary outcome for an arm.

    Examples:
    ## Example 1: Create a default Gaussian environment with custom random seed.
    >>> data = create_environment(env = "gaussian", random_seed = 42)

    ## Example 2: Create a default Bernoulli environment with custom random seed.
    >>> data = create_environment(env = "bernoulli", random_seed = 42)
    """

    environment = kwargs.get("env", None)
    n_arms = kwargs.get("n_arms", 2)
    arm_means = kwargs.get("arm_means", [[0.9, 0.1],[0.1, 0.9]])
    random_seed = kwargs.get("random_seed", None)
    observations = kwargs.get("observations", 2000)
    
    np.random.seed(random_seed)
    
    if environment == "bernoulli":
        bernoulli_data = np.vstack([np.hstack([np.random.choice([0, 1], size = (observations // len(arm_means), 1), p= [1-arm_means[_][i],arm_means[_][i]]) for i in range(n_arms)]) for _ in range(len(arm_means))])
        return bernoulli_data

