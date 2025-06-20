#%%
import os
from tqdm import tqdm
import numpy as np
import gymnasium as gym
from model import (ReplayBuffer, 
                   DQN)
from torch.utils.tensorboard import SummaryWriter

SEED = 123 
ENV_NAME = "CartPole-v1"
EPISODES = int(1e06)
BATCH_SIZE = 64
MAX_SIZE = int(1e06)
EPSILON = 1
EPSILON_DECAY = 0.9999
EPSILON_FINAL = 0.05
GAMMA = 0.99
LR = 1e-03
POLICY = 'mlp'
UPDATE_FREQUENCY = 4

env = gym.make(ENV_NAME)
NUM_ACTIONS = env.action_space.n
STATE_DIM = env.observation_space.shape

buffer = ReplayBuffer(batch_size=BATCH_SIZE,
                      max_size=MAX_SIZE)

model = DQN(state_dim=STATE_DIM,
            num_actions=NUM_ACTIONS,
            epsilon=EPSILON,
            epsilon_decay = EPSILON_DECAY,
            epsilon_final = EPSILON_FINAL,
            gamma=GAMMA,
            lr = LR,
            policy=POLICY)

writer = SummaryWriter(os.path.join(os.path.dirname(__file__), 'runs'))

total_steps = 0
prev_cum_rew = float('-inf')
eps = None
loss = None
for episode in tqdm(range(EPISODES)):
    state, _ = env.reset(seed=SEED)
    done = False
    cum_rew = 0
    while not done:
        action = model.epsilon_greedy(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        buffer.add([state, action, reward, next_state, terminated])
        if buffer.size() >= BATCH_SIZE and total_steps % UPDATE_FREQUENCY == 0:
            sampled_batch = buffer.sample()
            loss, eps = model.update(sampled_batch, BATCH_SIZE, episode)
        state = next_state
        cum_rew += reward
        done = terminated or truncated
        total_steps += 1
    if cum_rew >= prev_cum_rew:
        prev_cum_rew = cum_rew
        model.save_model(ENV_NAME)
        print(f"saved model at episode: {episode} with cum_rew {prev_cum_rew}")
    writer.add_scalar('total_reward_per_episode', cum_rew, episode)
    if eps is not None:
        writer.add_scalar('epsilon_per_episode', eps, episode)
    if loss is not None:
        writer.add_scalar('mse_loss_per_episode', np.array(loss), episode)
env.close() 