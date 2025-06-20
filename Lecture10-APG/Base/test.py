#%%
import os
import gymnasium as gym
import tensorflow as tf

SEED = 123
ENV_NAME = "HalfCheetah-v5"
EPISODES = 10

env = gym.make(ENV_NAME,
               render_mode = 'human')
NUM_ACTIONS = env.action_space.shape[0]

path = os.path.join(os.path.dirname(__file__), 'trained_models')
model = tf.keras.models.load_model(os.path.join(path, f'PPO_{ENV_NAME}.h5'))

for episode in range(EPISODES):
    state, _ = env.reset(seed = SEED)
    done = False
    while not done:
        out = model.predict(tf.convert_to_tensor([state], dtype=tf.float32), verbose = 0)
        mean = out[0][:NUM_ACTIONS]
        std = tf.nn.softplus(out[0][NUM_ACTIONS:])
        action =  mean + std * tf.random.normal((NUM_ACTIONS,))
        next_state, reward, terminated, truncated, info = env.step(action)
        state = next_state
        done = terminated or truncated
env.close()