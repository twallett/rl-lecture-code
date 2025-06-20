#%%
import os
from collections import deque
import random
import numpy as np
from tensorflow.keras import models, layers, optimizers, Sequential

class ReplayBuffer:
    def __init__(self, batch_size = 32, max_size = 1000000):
        self.buffer = deque(maxlen=max_size)
        self.batch_size = batch_size
        self.max_size = max_size
    
    def add(self, sampled_states):
        self.buffer.append(sampled_states)
        
    def sample(self):
        return random.sample(self.buffer, self.batch_size)
    
    def size(self):
        return len(self.buffer)
    
class DQN:
    
    def __init__(self, state_dim, num_actions, epsilon, epsilon_decay, epsilon_final, lr, gamma, policy = None):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.gamma = gamma
        if policy == 'cnn':
            self.q = Sequential([
                layers.Conv2D(32, (8,8), strides = 4, activation = 'relu', input_shape = state_dim),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (4,4), strides = 2, activation = 'relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3,3), strides = 1, activation = 'relu'),
                layers.Flatten(),
                layers.Dense(512, activation = 'relu'),
                layers.Dense(num_actions, activation = 'linear')
            ])
        elif policy == 'mlp':
            self.q = Sequential([
                layers.Dense(64, activation = 'relu', input_shape = state_dim),
                layers.Dense(64, activation = 'relu'),
                layers.Dense(num_actions, activation = 'linear')
            ])
        self.q.compile(optimizer = optimizers.Adam(learning_rate = lr), loss = 'mse')
        self.fixed_q = models.clone_model(self.q)
        self.fixed_q.set_weights(self.q.get_weights())
        self.t = 0
        self.epsilon_decay = epsilon_decay
        self.epsilon_final = epsilon_final
        self.prev_episode = float('-inf')
    
    def epsilon_greedy(self, state):
        q_values = self.q.predict(np.expand_dims(state, axis = 0), verbose = 0)
        greedy_action = np.argmax(q_values)
        behavior_probabilities = np.ones(self.num_actions) * (self.epsilon / self.num_actions)
        behavior_probabilities[greedy_action] += (1 - self.epsilon)
        selected_action = np.random.choice(np.arange(self.num_actions), p=behavior_probabilities)
        return selected_action
        
    def update(self, sampled_batch, batch_size, episode):
        if len(sampled_batch) < batch_size:
            self.t += 1
            return
        states = np.array([sample[0] for sample in sampled_batch])
        actions = np.array([sample[1] for sample in sampled_batch])
        rewards = np.array([sample[2] for sample in sampled_batch])
        next_states = np.array([sample[3] for sample in sampled_batch])
        terminated = np.array([sample[4] for sample in sampled_batch])
        q_targets = self.q.predict(states, verbose = 0)
        for i in range(len(sampled_batch)):
            if terminated[i]:
                q_targets[i,actions[i]] = rewards[i]
            else:
                fixed_q = self.fixed_q.predict(np.expand_dims(next_states[i,:],axis=0), verbose = 0)
                q_targets[i,actions[i]] = rewards[i] + self.gamma * np.max(fixed_q)
        history = self.q.fit(states, q_targets, batch_size = batch_size, epochs = 1, verbose = 0)
        self.t += 1
        if self.t % 100 == 0:
            self.fixed_q.set_weights(self.q.get_weights())
        if episode > self.prev_episode:
            self.prev_episode = episode
            self.epsilon = max(self.epsilon_final, self.epsilon * self.epsilon_decay)
        return history.history['loss'][0], self.epsilon
            
    def save_model(self, env_name):
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_path, 'trained_models')
        os.makedirs(model_path, exist_ok=True)
        self.q.save(os.path.join(model_path, f'DQN_{env_name}.h5'))

#%%
