#%%
import os
from tqdm import tqdm
from model import (Node, 
                   MCTS)
import gymnasium as gym 
from torch.utils.tensorboard import SummaryWriter

SEED = 123 
ENV_NAME = 'CartPole-v1'
EPISODES = 1
SIMULATIONS = 50
ITERATIONS = 200
MAX_DEPTH = 50
C = 2
GAMMA = 0.99

env = gym.make(ENV_NAME)
NUM_ACTIONS = env.action_space.n

writer = SummaryWriter(os.path.join(os.path.dirname(__file__), 'runs'))

model = MCTS(env_name=ENV_NAME,
             simulations = SIMULATIONS,
             iterations=ITERATIONS,
             c = C,
             max_depth=MAX_DEPTH,
             gamma=GAMMA)
            
for episode in tqdm(range(EPISODES)):
    state, _ = env.reset(seed = SEED)
    if episode == 0:
        root_state = state
        node = Node(state, NUM_ACTIONS)
    done = False
    unwrapped_env = env.unwrapped
    unwrapped_env.state = root_state
    cum_rew = 0
    while not done:
        action, node = model.update(node)
        state, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            node = model.return_to_root(node)
        cum_rew += reward
        done = terminated or truncated
        print(cum_rew)
    writer.add_scalar('total_reward_per_episode', cum_rew, episode)
env.close()
