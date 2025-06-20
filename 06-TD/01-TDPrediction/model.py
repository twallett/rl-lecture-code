#%%
import numpy as np

class TDPrediction:

    def __init__(self, num_states, num_actions, alpha = 0.1, gamma = 0.9, seed = 123):
        np.random.seed(seed)
        self.v = np.zeros(num_states)
        self.num_actions = num_actions
        self.gamma = gamma
        self.alpha = alpha
    
    def pi(self):
        return np.random.randint(self.num_actions)
    
    def update(self, state, next_state, reward):
        self.v[state] += self.alpha * (reward + (self.gamma * self.v[next_state]) - self.v[state]) 
        return self.v