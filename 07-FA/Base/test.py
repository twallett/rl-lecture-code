#%%
import os
import pickle
import numpy as np
import gymnasium as gym
from tile_coding import *

SEED = 123
ENV_NAME = "MountainCar-v0"
EPISODES = 10
NUM_TILINGS = 8
FPS = 100

env = gym.make(ENV_NAME,
               render_mode = 'human')

env.metadata['render_fps'] = FPS
NUM_ACTIONS = env.action_space.n
STATE_DIM = env.observation_space.shape
HIGH_BOUND = env.observation_space.high
LOW_BOUND = env.observation_space.low

base_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_path, 'trained_models')
path_weights = os.path.join(model_path, f'SGSARSA_{ENV_NAME}_weights.pkl')
weights = pickle.load(open(path_weights, 'rb'))
path_tiles = os.path.join(model_path, f'SGSARSA_{ENV_NAME}_tiles.pkl')
tile_coding = pickle.load(open(path_tiles, 'rb'))

def phi(state):
    normalized_state = np.zeros(STATE_DIM)
    for enum, (bound_max, bound_min) in enumerate(zip(HIGH_BOUND, LOW_BOUND)):
        if bound_max == float('inf') or bound_min == float('-inf'):
            bound_max = np.tanh(bound_max)
            bound_min = np.tanh(bound_min)
        normalized_state[enum] = (NUM_TILINGS * state[enum] / (bound_max - bound_min))
    indxs = []
    for action in range(NUM_ACTIONS):
        indxs.append( np.array(tiles(tile_coding, NUM_TILINGS, normalized_state, [action])) )
    one_hot = np.zeros((NUM_TILINGS**4, NUM_ACTIONS))
    for action in range(NUM_ACTIONS):
        indx = indxs[action]
        for i in indx:
            one_hot[i, action] = 1
    return one_hot

for episode in range(EPISODES):
    state, _ = env.reset(seed=SEED)
    done = False
    while not done:
        x = phi(state)
        action = np.argmax(weights.T @ x)
        state, reward, terminated, truncated, info = env.step(action)
        dones = terminated or truncated
        done = dones
env.close()