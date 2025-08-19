#%%
import os
import gymnasium as gym
import tensorflow as tf

SEED = 123
ENV_NAME = "CartPole-v1"
EPISODES = 10

env = gym.make(ENV_NAME,
               render_mode = 'human')
NUM_ACTIONS = env.action_space.n

path = os.path.join(os.path.dirname(__file__), 'trained_models')
model = tf.keras.models.load_model(os.path.join(path, f'PG_{ENV_NAME}.h5'))

for episode in range(EPISODES):
    state, _ = env.reset(seed = SEED)
    done = False
    while not done:
        logits = model.predict(tf.convert_to_tensor([state], dtype=tf.float32))
        action_dist = tf.random.categorical(logits, num_samples=1)
        action = tf.squeeze(action_dist, axis = -1).numpy().item()
        next_state, reward, terminated, truncated, info = env.step(action)
        state = next_state
        done = terminated or truncated
env.close()
