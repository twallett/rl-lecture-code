#%%
import numpy as np

class ExploringStarts:
    
    def __init__(self, num_states, num_actions, gamma = 0.9, seed = 123):
        np.random.seed(seed)
        self.q = np.zeros((num_states, num_actions))
        self.num_actions = num_actions
        self.gamma = gamma
        self.returns = {f"{state},{action}": [] for state in range(num_states) for action in range(num_actions)}
    
    def pi(self):
        return np.random.randint(self.num_actions)
    
    def update(self, sampled_states, sampled_actions, sampled_rewards, t):
        g = 0
        for i in reversed(range(0, t)):
            state = sampled_states[i]
            action = sampled_actions[i]
            reward = sampled_rewards[i]
            g = (self.gamma * g) + reward
            if not any(state == repeat and action == sampled_actions[j] for j, repeat in enumerate(sampled_states[:i])):
                self.returns[f"{state},{action}"].append(g)
                self.q[state, action] = np.mean(self.returns[f"{state},{action}"])
        return self.q
