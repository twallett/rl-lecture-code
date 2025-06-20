#%%
import numpy as np

class MCPrediction:
    
    def __init__(self, num_states, num_actions, gamma = 0.9, seed = 123):
        np.random.seed(seed)
        self.v = np.zeros(num_states)
        self.num_actions = num_actions
        self.gamma = gamma
        self.returns = {state: [] for state in range(num_states)}
    
    def pi(self):
        return np.random.randint(self.num_actions)
    
    def update(self, sampled_states, sampled_rewards, t):
        g = 0
        for i in reversed(range(0, t)):
            state = sampled_states[i]
            reward = sampled_rewards[i]
            g = (self.gamma * g) + reward
            if not any(state == repeat for repeat in sampled_states[:i]):
                self.returns[state].append(g)
                self.v[state] = np.mean(self.returns[state])
        return self.v
