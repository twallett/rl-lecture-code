#%%
import os
from tqdm import tqdm
import gymnasium as gym
from model import SemiGradientSARSA
from torch.utils.tensorboard import SummaryWriter

SEED = 123
ENV_NAME = "MountainCar-v0"
EPISODES = 5000
EPSILON = 1
EPSILON_DECAY = 0.9995
EPSILON_FINAL = 0.01
ALPHA = 2e-02
GAMMA = 0.99
NUM_TILINGS = 8

env = gym.make(ENV_NAME)

NUM_ACTIONS = env.action_space.n
STATE_DIM = env.observation_space.shape
HIGH_BOUND = env.observation_space.high
LOW_BOUND = env.observation_space.low

model = SemiGradientSARSA(state_dim=STATE_DIM,
                          num_actions=NUM_ACTIONS,
                          epsilon=EPSILON,
                          epsilon_decay=EPSILON_DECAY,
                          epsilon_final=EPSILON_FINAL,
                          alpha=ALPHA,
                          high_bound = HIGH_BOUND,
                          low_bound = LOW_BOUND,
                          num_tilings=NUM_TILINGS, 
                          gamma=GAMMA)

writer = SummaryWriter(os.path.join(os.path.dirname(__file__), 'runs'))

prev_cum_rew = float('-inf')
for episode in tqdm(range(EPISODES)):
    state, _ = env.reset(seed=SEED)
    action = model.epsilon_greedy(state)
    done = False
    cum_rew = 0
    while not done:
        next_state, reward, terminated, truncated, info = env.step(action)
        dones = terminated or truncated
        if dones:
            model.update(state, action, reward, None, None, dones, episode)
        else:
            next_action = model.epsilon_greedy(next_state)
            model.update(state, action, reward, next_state, next_action, dones, episode)
            state = next_state
            action = next_action 
        cum_rew += reward
        done = dones
    if cum_rew >= prev_cum_rew:
        prev_cum_rew = cum_rew
        model.save_model(ENV_NAME)
        print(f"saved model at episode: {episode} with cum_rew {prev_cum_rew}")
    writer.add_scalar('total_reward_per_episode', cum_rew, episode)
env.close()