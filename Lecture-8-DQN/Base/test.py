import os
import gymnasium as gym
import tensorflow as tf
import numpy as np
from keras.metrics import MeanSquaredError

SEED = 123 
ENV_NAME = "CartPole-v1"
EPISODES = 10

env = gym.make(ENV_NAME,
               render_mode = 'human')

path = os.path.join(os.path.dirname(__file__), 'trained_models')
model = tf.keras.models.load_model(os.path.join(path, f'DQN_{ENV_NAME}.h5'), custom_objects={'mse': MeanSquaredError})

for episode in range(EPISODES):
    state, _ = env.reset(seed=SEED)
    done = False
    while not done:
        action = np.argmax(model.predict(np.expand_dims(state, axis=0), verbose = 0))
        next_state, reward, terminated, truncated, info = env.step(action)
        state = next_state
        done = terminated or truncated
env.close()
