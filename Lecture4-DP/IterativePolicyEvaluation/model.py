#%%
import numpy as np

class IterativePolicyEvaluation:
    
    def __init__(self, num_states, gamma = 0.9, seed = 123):
        np.random.seed(seed)
        self.gamma = gamma
        self.v = np.zeros((num_states))

    def update(self, state, next_states, rewards):
        v = []
        for next_state, reward in zip(next_states, rewards):
            v.append(0.25 * 1 * (reward + (self.gamma * self.v[next_state])) )
        self.v[state] = np.sum(v)
        return self.v