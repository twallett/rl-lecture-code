#%%
import os
import pickle
import numpy as np
from tile_coding import *

class SemiGradientSARSA:
    
    def __init__(self, state_dim, num_actions, num_tilings, high_bound, low_bound, epsilon, epsilon_decay, epsilon_final, alpha, gamma):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.high_bound = high_bound
        self.low_bound = low_bound
        self.epsilon_decay = epsilon_decay
        self.epsilon_final = epsilon_final
        self.num_tilings = num_tilings
        self.tile_coding = IHT(num_tilings**4)
        self.weights = np.random.uniform(low = -0.05, high = 0.05, size = (num_tilings**4, 1))
        self.prev_episode = float('-inf')
        
    def phi(self, state):
        normalized_state = np.zeros(self.state_dim)
        for enum, (bound_max, bound_min) in enumerate(zip(self.high_bound, self.low_bound)):
            if bound_max == float('inf') or bound_min == float('-inf'):
                bound_max = np.tanh(bound_max)
                bound_min = np.tanh(bound_min)
            normalized_state[enum] = (self.num_tilings * state[enum]) / (bound_max - bound_min)
        indxs = []
        for action in range(self.num_actions):
            indxs.append( np.array(tiles(self.tile_coding, self.num_tilings, normalized_state, [action])) )
        one_hot = np.zeros((self.num_tilings**4, self.num_actions))
        for action in range(self.num_actions):
            indx = indxs[action]
            for i in indx:
                one_hot[i, action] = 1
        return one_hot
        
    def epsilon_greedy(self, state):
        phi_s_a = self.phi(state)
        q_values = (self.weights.T @ phi_s_a).squeeze()
        greedy_action = np.argmax(q_values)
        behavior_probabilities = np.ones(self.num_actions) * (self.epsilon / self.num_actions)
        behavior_probabilities[greedy_action] += (1 - self.epsilon)
        selected_action = np.random.choice(np.arange(self.num_actions), p=behavior_probabilities)
        return selected_action

    def update(self, state, action, reward, next_state, next_action, dones, episode):
        phi_s_a = self.phi(state)
        if dones:
            td_error = reward - (self.weights.T @ phi_s_a[:, action])
        else:
            phi_s_a_next = self.phi(next_state)
            td_target = reward + self.gamma * (self.weights.T @ phi_s_a_next[:, next_action])
            td_error = td_target - (self.weights.T @ phi_s_a[:, action])
        gradient = phi_s_a[:, action].reshape(-1, 1)
        self.weights += self.alpha * td_error * gradient
        if episode > self.prev_episode:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_final)

    def save_model(self, env_name):
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_path, 'trained_models')
        os.makedirs(model_path, exist_ok=True)
        path_weights = os.path.join(model_path, f'SGSARSA_{env_name}_weights.pkl')
        pickle.dump(self.weights, open(path_weights, 'wb') )
        path_tiles = os.path.join(model_path, f'SGSARSA_{env_name}_tiles.pkl')
        pickle.dump(self.tile_coding, open(path_tiles, 'wb'))
        