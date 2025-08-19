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
            tf.convert_to_tensor(list(actions), dtype=tf.int32),
            tf.convert_to_tensor(list(rewards), dtype=tf.float32)
        )

    def add_trajectory(self):
        self.buffer.append((list(self.states), list(self.actions), list(self.rewards)))
    
    def reset(self):
        self.states = deque()
        self.actions = deque()
        self.rewards = deque()

class PPOClip:
    def __init__(self, state_dim, num_actions, gamma, alpha_pi, alpha_v, epsilon, policy='mlp'):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon

        if policy == 'mlp':
            self.pi = Sequential([
                layers.Input(shape=state_dim),
                layers.Dense(64, activation='tanh'),
                layers.Dense(64, activation='tanh'),
                layers.Dense(num_actions),  # logits
            ])
            self.pi_k = Sequential([
                layers.Input(shape=state_dim),
                layers.Dense(64, activation='tanh'),
                layers.Dense(64, activation='tanh'),
                layers.Dense(num_actions),
            ])
            self.pi_k.set_weights(self.pi.get_weights())
            
            self.v = Sequential([
                layers.Input(shape=state_dim),
                layers.Dense(64, activation='tanh'),
                layers.Dense(64, activation='tanh'),
                layers.Dense(1),
            ])
            self.optimizer_pi = optimizers.Adam(learning_rate=alpha_pi)
            self.optimizer_v = optimizers.Adam(learning_rate=alpha_v)

    def policy(self, state):
        logits = self.pi(tf.convert_to_tensor([state], dtype=tf.float32))
        action_dist = tf.random.categorical(logits, num_samples=1)
        return tf.squeeze(action_dist, axis = -1).numpy().item()

    def compute_advantage(self, sampled_states, sampled_rewards):
        rewards_to_go, values, adv = [], [], []
        g = 0 
        for state, reward in list(zip(sampled_states, sampled_rewards))[::-1]:
            g = self.gamma * g + reward
            rewards_to_go.append(g)
            value = self.v(tf.convert_to_tensor([state], dtype=tf.float32))[0, 0]
            values.append(value)
            adv.append(g - value)
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

        states_tensor = tf.convert_to_tensor(np.array(sampled_states), dtype=tf.float32)
        actions_tensor = tf.convert_to_tensor(sampled_actions, dtype=tf.int32)
        adv_tensor = tf.convert_to_tensor(adv, dtype=tf.float32)
        rewards_to_go_tensor = tf.convert_to_tensor(rewards_to_go, dtype=tf.float32)

        with tf.GradientTape() as tape:
            logits = self.pi(states_tensor)
            old_logits = self.pi_k(states_tensor)
            log_probs = -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=actions_tensor)
            old_log_probs = -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=old_logits, labels=actions_tensor)

            ratio = tf.exp(log_probs - old_log_probs)
            clipped_ratio = tf.clip_by_value(ratio, 1 - self.epsilon, 1 + self.epsilon)
            policy_loss = -tf.reduce_mean(tf.minimum(ratio * adv_tensor, clipped_ratio * adv_tensor))

        gradients = tape.gradient(policy_loss, self.pi.trainable_variables)
        self.optimizer_pi.apply_gradients(zip(gradients, self.pi.trainable_variables))

        with tf.GradientTape() as tape:
            value_preds = tf.squeeze(self.v(states_tensor))
            value_loss = tf.reduce_mean((rewards_to_go_tensor - value_preds) ** 2)
        gradients = tape.gradient(value_loss, self.v.trainable_variables)
        self.optimizer_v.apply_gradients(zip(gradients, self.v.trainable_variables))

        self.pi_k.set_weights(self.pi.get_weights())

    def save_model(self, env_name):
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_path, 'trained_models')
        os.makedirs(model_path, exist_ok=True)
        self.pi.save(os.path.join(model_path, f'PPO_{env_name}.h5'))