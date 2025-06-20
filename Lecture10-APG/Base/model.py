#%%
import os
from collections import deque
import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential, layers, optimizers

class Buffer:
    
    def __init__(self, max_trajectories):
        self.buffer = deque(maxlen=max_trajectories)
    
    def sample_state(self, state):
        self.states.append(state)
    
    def sample_action(self, action):
        self.actions.append(action)
    
    def sample_reward(self, reward):
        self.rewards.append(reward)
    
    def get_batch(self, idx):
        batch = self.buffer[idx]
        states, actions, rewards = batch
        return (
            tf.convert_to_tensor(list(states), dtype=tf.float32),
            tf.convert_to_tensor(list(actions), dtype=tf.float32),
            tf.convert_to_tensor(list(rewards), dtype=tf.float32)
        )

    def add_trajectory(self):
        self.buffer.append((list(self.states), list(self.actions), list(self.rewards)))
    
    def reset(self):
        self.states = deque()
        self.actions = deque()
        self.rewards = deque()

class PPOClip:
    
    def __init__(self, state_dim, num_actions, gamma, alpha_pi, alpha_v, epsilon, policy = None):
        self.state_dim = state_dim
        self.num_actions = num_actions
        if policy == 'mlp':
            self.pi = Sequential([
                layers.Dense(64, activation = 'tanh', input_shape = state_dim),
                layers.Dense(64, activation = 'tanh'),
                layers.Dense(num_actions * 2),
            ])
            self.pi_k = Sequential([
                layers.Dense(64, activation='tanh', input_shape=state_dim),
                layers.Dense(64, activation='tanh'),
                layers.Dense(num_actions * 2),
            ])
            self.pi_k.set_weights(self.pi.get_weights())
            self.v = Sequential([
                layers.Dense(64, activation = 'tanh', input_shape = state_dim),
                layers.Dense(64, activation = 'tanh'),
                layers.Dense(1),
            ])
            self.optimizer_pi = optimizers.Adam(learning_rate = alpha_pi)
            self.optimizer_v = optimizers.Adam(learning_rate = alpha_v)
        self.epsilon = epsilon
        self.gamma = gamma

    def policy(self, state):
        out = self.pi(tf.convert_to_tensor([state], dtype=tf.float32))
        mean = out[0][:self.num_actions]
        std = tf.nn.softplus(out[0][self.num_actions:])
        return mean + std * tf.random.normal((self.num_actions,))

    def compute_advantage(self, sampled_states, sampled_rewards):
        rewards_to_go, values, adv = [], [], []
        g = 0 
        for state, reward in list(zip(sampled_states, sampled_rewards))[::-1]:
            g = self.gamma * g + reward
            rewards_to_go.append(g)
            value = self.v(tf.convert_to_tensor([state], dtype=tf.float32))
            values.append(value[0][0])
            a = g - value[0][0]
            adv.append(a)
        return np.array(rewards_to_go[::-1]), np.array(values[::-1]), np.array(adv[::-1])
    
    def update(self, buffer):
        sampled_states, sampled_actions, sampled_rewards = [], [], []
        for _ in range(len(buffer.buffer)):
            states, actions, rewards = buffer.get_batch(0)
            sampled_states.extend(states)
            sampled_actions.extend(actions)
            sampled_rewards.extend(rewards)
            buffer.buffer.popleft()
        rewards_to_go, values, adv = self.compute_advantage(sampled_states, sampled_rewards)
        with tf.GradientTape() as tape:
            sampled_actions = tf.convert_to_tensor(sampled_actions, dtype=tf.float32)
            adv = tf.convert_to_tensor(adv, dtype=tf.float32)
            out_pi = self.pi(tf.convert_to_tensor(sampled_states, dtype=tf.float32))
            mean, std = out_pi[:, :self.num_actions], tf.nn.softplus(out_pi[:, self.num_actions:])
            log_probs = -0.5 * tf.reduce_sum(((sampled_actions - mean) ** 2) / (std ** 2) + 2 * tf.math.log(std) + tf.math.log(2 * np.pi), axis=1)
            out_pi_k = self.pi_k(tf.convert_to_tensor(sampled_states, dtype=tf.float32))
            mean_k, std_k = out_pi_k[:, :self.num_actions], tf.nn.softplus(out_pi_k[:, self.num_actions:])
            log_probs_k = -0.5 * tf.reduce_sum(((sampled_actions - mean_k) ** 2) / (std_k ** 2) + 2 * tf.math.log(std_k) + tf.math.log(2 * np.pi), axis=1)
            ratio = tf.exp(log_probs - log_probs_k)
            clipped_ratio = tf.clip_by_value(ratio, 1 - self.epsilon, 1 + self.epsilon)
            loss = -tf.reduce_mean(tf.minimum(ratio * adv, clipped_ratio * adv))
        gradients = tape.gradient(loss, self.pi.trainable_variables)
        self.optimizer_pi.apply_gradients(zip(gradients, self.pi.trainable_variables))
        with tf.GradientTape() as tape:
            values = tf.squeeze(self.v(tf.convert_to_tensor(sampled_states, dtype=tf.float32)))
            value_loss = tf.reduce_mean((rewards_to_go - values) ** 2)
        gradients = tape.gradient(value_loss, self.v.trainable_variables)
        self.optimizer_v.apply_gradients(zip(gradients, self.v.trainable_variables))
        self.pi_k.set_weights(self.pi.get_weights())
        
    def save_model(self, env_name):
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_path, 'trained_models')
        os.makedirs(model_path, exist_ok=True)
        self.pi.save(os.path.join(model_path, f'PPO_{env_name}.h5'))