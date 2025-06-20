#%%
import numpy as np
import pandas as pd

class EpsilonGreedy:
    """    Implementation of the Epsilon-Greedy algorithm for multi-armed bandits.

    Parameters:
    - n_arms (int): The number of arms in the bandit.
    - epsilon (float): Exploration-exploitation trade-off parameter (default is 0).
    - manual_prior (tuple): Tuple containing the initial values for Q (action-value estimates) and N (action counts).
      If None, default values of 0 are used for Q and N.
    - random_seed (int): Seed for the random number generator (default is None).

    Methods:
    - bandit(data, A, t): Simulates pulling an arm and returns the corresponding reward.
    - action(): Chooses an action based on the Epsilon-Greedy strategy.
    - update(A, R): Updates action-value estimates and action counts based on the received reward.
    - train(data): Trains the Epsilon-Greedy algorithm on a given dataset.
    - create_table(): Generates a table with information about arms, action counts, and estimated rewards.
    - save(): Saves the current state of action-value estimates and action counts to a DataFrame.
    - recommend(): Recommends an action based on the current estimates (for exploration).
    - reset(): Resets the action-value estimates and action counts to zeros.

    Attributes:
    - n_arms (int): The number of arms in the bandit.
    - epsilon (float): Exploration-exploitation trade-off parameter.
    - Q (numpy.ndarray): Action-value estimates for each arm.
    - N (numpy.ndarray): Action counts for each arm.
    - rewards_list (list): List to store received rewards during training.
    - rewards_matrix (numpy.ndarray): Matrix to store rewards for each arm at each time step.
    """

    def __init__(self, n_arms, epsilon = 0, manual_prior = None, random_seed = None):
        np.random.seed(random_seed)
        self.n_arms = n_arms  
        if manual_prior:
            self.Q = np.ones((n_arms,1)) * manual_prior[0]
            self.N = np.ones((n_arms,1)) * manual_prior[1]
        else:
            self.Q = np.zeros((n_arms, 1))
            self.N = np.zeros((n_arms, 1))
        self.epsilon = epsilon
        self.rewards_list = []
        self.rewards_matrix = np.zeros((1, n_arms))

    def bandit(self, data, A, t):
        """
        The function bandit() is assumed to take an action and return a corresponding reward.
        """
        rewards_pull = data.copy()[t:t+1][0] 
        R = rewards_pull[A] 
        rewards_pull[:A] = 0
        rewards_pull[A+1:] = 0
        self.rewards_matrix = np.vstack([self.rewards_matrix, rewards_pull.reshape((1,-1))]) 
        self.rewards_list.append(R) 
        
        return R

    def action(self):
        if np.random.uniform(0, 1) <= self.epsilon: 
            # Action - exploring
            return np.random.randint(0, self.n_arms)
        else:
            # Action - exploiting
            return np.argmax(self.Q)
    
    def update(self, A, R):
        self.N[A] += 1 
        self.Q[A] += (1 / self.N[A]) * (R - self.Q[A]) 
    
    def train(self, data):

        for t in range(len(data)):
            
            A = self.action()
            
            R = self.bandit(data, A, t)
            
            self.update(A, R)

        table = self.create_table()
        
        return table, self.rewards_list, self.rewards_matrix[1:,:]
    
    def create_table(self):
        table = np.hstack([np.arange(1, self.n_arms + 1).reshape(self.n_arms, 1),
                           self.N, self.Q.round(2)]).astype(float)
        table = pd.DataFrame(data=table, columns=["Arms", "Arm Selection", "E(reward|action)"])
        table = table.to_string(index=False)
        return table
    
    def save(self):
        df = pd.DataFrame({"N": self.N.flatten(), 'Q': self.Q.flatten()})
        return df

    def reset(self):
        self.Q = np.zeros((self.n_arms, 1))
        self.N = np.zeros((self.n_arms, 1))
        self.rewards_list = []
        self.rewards_matrix = np.zeros((1, self.n_arms))