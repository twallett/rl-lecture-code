#%%
import numpy as np

def create_environment(**kwargs):
    """
    Create synthetic data for a multi-armed bandit environment.

    Parameters:
    - env (str): The type of environment to generate ("gaussian" or "bernoulli").
    - n_arms (int): The number of arms in the bandit (default is 2).
    - arm_means (list): List of lists representing mean values for each arm in each environment.
      For "gaussian" environment, it should be a list of lists of floats.
      For "bernoulli" environment, it should be a list of lists of probabilities.
      Default is [[0.9, 0.1], [0.1, 0.9]].
    - arm_stds (list): List of lists representing standard deviations for each arm in each environment.
      Only used for "gaussian" environment. If not provided, defaults to 0.1 for all arms.
      Should have the same structure as arm_means.
    - random_seed (int): Seed for the random number generator (default is None).
    - observations (int): Number of observations to generate for each arm (default is 2000).

    Returns:
    - numpy.ndarray: Synthetic data matrix representing observations from the specified multi-armed bandit environment.
      Shape: (total_observations, n_arms) where total_observations = observations
      For "gaussian" environment, each column corresponds to continuous rewards for an arm.
      For "bernoulli" environment, each column corresponds to binary outcomes for an arm.

    Examples:
    ## Example 1: Create a default Gaussian environment with custom random seed.
    >>> data = create_environment(env="gaussian", random_seed=42)

    ## Example 2: Create a Gaussian environment with custom means and standard deviations.
    >>> data = create_environment(
    ...     env="gaussian", 
    ...     n_arms=3,
    ...     arm_means=[[0.5, 0.7, 0.3], [0.8, 0.4, 0.6]], 
    ...     arm_stds=[[0.1, 0.15, 0.2], [0.1, 0.1, 0.1]],
    ...     observations=1000,
    ...     random_seed=42
    ... )

    ## Example 3: Create a default Bernoulli environment with custom random seed.
    >>> data = create_environment(env="bernoulli", random_seed=42)

    ## Example 4: Create a Bernoulli environment with custom probabilities.
    >>> data = create_environment(
    ...     env="bernoulli", 
    ...     n_arms=4,
    ...     arm_means=[[0.2, 0.5, 0.8, 0.3], [0.7, 0.1, 0.9, 0.4]], 
    ...     observations=1500,
    ...     random_seed=42
    ... )
    """

    environment = kwargs.get("env", None)
    n_arms = kwargs.get("n_arms", 2)
    arm_means = kwargs.get("arm_means", [[0.9, 0.1], [0.1, 0.9]])
    arm_stds = kwargs.get("arm_stds", None)
    random_seed = kwargs.get("random_seed", None)
    observations = kwargs.get("observations", 2000)
    
    # Input validation
    if environment not in ["gaussian", "bernoulli"]:
        raise ValueError("Environment must be either 'gaussian' or 'bernoulli'")
    
    if len(arm_means) == 0:
        raise ValueError("arm_means cannot be empty")
    
    if not all(len(env_means) == n_arms for env_means in arm_means):
        raise ValueError(f"Each environment in arm_means must have exactly {n_arms} arms")
    
    # Set random seed
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n_environments = len(arm_means)
    obs_per_env = observations // n_environments
    
    if environment == "gaussian":
        # Set default standard deviations if not provided
        if arm_stds is None:
            arm_stds = [[0.1] * n_arms for _ in range(n_environments)]
        
        # Validate arm_stds structure
        if len(arm_stds) != n_environments:
            raise ValueError("arm_stds must have the same number of environments as arm_means")
        
        if not all(len(env_stds) == n_arms for env_stds in arm_stds):
            raise ValueError(f"Each environment in arm_stds must have exactly {n_arms} arms")
        
        # Generate Gaussian data
        gaussian_data = []
        for env_idx in range(n_environments):
            env_data = []
            for arm_idx in range(n_arms):
                mean = arm_means[env_idx][arm_idx]
                std = arm_stds[env_idx][arm_idx]
                arm_data = np.random.normal(mean, std, size=(obs_per_env, 1))
                env_data.append(arm_data)
            
            # Concatenate arms horizontally for this environment
            env_data_matrix = np.hstack(env_data)
            gaussian_data.append(env_data_matrix)
        
        # Stack all environments vertically
        return np.vstack(gaussian_data)
    
    elif environment == "bernoulli":
        # Validate probabilities for Bernoulli
        for env_idx, env_means in enumerate(arm_means):
            for arm_idx, prob in enumerate(env_means):
                if not (0 <= prob <= 1):
                    raise ValueError(f"Bernoulli probabilities must be between 0 and 1. "
                                   f"Found {prob} at environment {env_idx}, arm {arm_idx}")
        
        # Generate Bernoulli data
        bernoulli_data = []
        for env_idx in range(n_environments):
            env_data = []
            for arm_idx in range(n_arms):
                prob = arm_means[env_idx][arm_idx]
                arm_data = np.random.choice([0, 1], size=(obs_per_env, 1), p=[1-prob, prob])
                env_data.append(arm_data)
            
            # Concatenate arms horizontally for this environment
            env_data_matrix = np.hstack(env_data)
            bernoulli_data.append(env_data_matrix)
        
        # Stack all environments vertically
        return np.vstack(bernoulli_data)
