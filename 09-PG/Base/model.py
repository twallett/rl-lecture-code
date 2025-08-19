#%%
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential, layers, optimizers

class PolicyGradient:
    
    def __init__(self, state_dim, num_actions, gamma, alpha_pi, alpha_v, policy = None):
        self.state_dim = state_dim
        self.num_actions = num_actions
        if policy == 'mlp':
            self.pi = Sequential([
                layers.Dense(64, activation = 'tanh', input_shape = state_dim),
                layers.Dense(64, activation = 'tanh'),
                layers.Dense(num_actions),
            ])
            self.v = Sequential([
                layers.Dense(64, activation = 'tanh', input_shape = state_dim),
                layers.Dense(64, activation = 'tanh'),
                layers.Dense(1),
            ])
            self.optimizer_pi = optimizers.Adam(learning_rate = alpha_pi)
            self.optimizer_v = optimizers.Adam(learning_rate = alpha_v)
        self.gamma = gamma

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
            value = self.v(tf.convert_to_tensor([state], dtype=tf.float32))
            values.append(value[0][0])
            a = g - value[0][0]
            adv.append(a)
        return (
            tf.convert_to_tensor(list(rewards_to_go[::-1]), dtype=tf.float32),
            tf.convert_to_tensor(list(values[::-1]), dtype=tf.float32),
            tf.convert_to_tensor(list(adv[::-1]), dtype=tf.float32)
        )
    
    def update(self, sampled_states, sampled_actions, sampled_rewards):
        
        rewards_to_go, values, adv = self.compute_advantage(sampled_states, sampled_rewards)
        
        with tf.GradientTape() as tape:
            logits = self.pi(tf.convert_to_tensor(sampled_states, dtype=tf.float32))
            log_probs = tf.convert_to_tensor([tf.nn.log_softmax(logits[i,:])[sampled_actions[i]] for i in range(len(sampled_actions))],dtype=tf.float32)
            loss = -tf.reduce_mean(log_probs * adv)
        gradients = tape.gradient(loss, self.pi.trainable_variables)
        self.optimizer_pi.apply_gradients(zip(gradients, self.pi.trainable_variables))
        
        states_tensor = tf.convert_to_tensor(np.array(sampled_states), dtype=tf.float32)
        with tf.GradientTape() as tape:
            value_preds = tf.squeeze(self.v(states_tensor))
            value_loss = tf.reduce_mean((rewards_to_go - value_preds) ** 2) 
        gradients = tape.gradient(value_loss, self.v.trainable_variables)
        self.optimizer_v.apply_gradients(zip(gradients, self.v.trainable_variables))
        
    def save_model(self, env_name):
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_path, 'trained_models')
        os.makedirs(model_path, exist_ok=True)
        self.pi.save(os.path.join(model_path, f'PG_{env_name}.h5'))
        