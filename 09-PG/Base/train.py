#%%
import os
from tqdm import tqdm
import gymnasium as gym 
# import highway_env
from model import PolicyGradient
from torch.utils.tensorboard import SummaryWriter

SEED = 123 
ENV_NAME = "CartPole-v1"
EPISODES = 1000
GAMMA = 0.99
ALPHA_PI = 3e-04
ALPHA_V = 1e-03
POLICY = 'mlp'

# highway_env._register_highway_envs()

env = gym.make(ENV_NAME)
ACTIONS_DIM = env.action_space.n
STATE_DIM = env.observation_space.shape

model = PolicyGradient(state_dim=STATE_DIM,
                       num_actions=ACTIONS_DIM,
                       gamma=GAMMA,
                       alpha_pi = ALPHA_PI,
                       alpha_v =ALPHA_V, 
                       policy=POLICY)

writer = SummaryWriter(os.path.join(os.path.dirname(__file__), 'runs'))

prev_cum_rew = float('-inf')
for episode in tqdm(range(EPISODES)):
    state, _ = env.reset(seed=SEED)
    done = False
    cum_rew = 0
    sampled_states = [state]
    sampled_actions = []
    sampled_rewards = []
    while not done:
        action = model.policy(state)
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if not done:
            sampled_states.append(state)
        sampled_actions.append(action)
        sampled_rewards.append(reward)
        cum_rew += reward
    model.update(sampled_states, sampled_actions, sampled_rewards)
    if cum_rew >= prev_cum_rew:
        prev_cum_rew = cum_rew
        model.save_model(ENV_NAME)
        print(f"saved model at episode: {episode} with cum_rew {prev_cum_rew}")
    writer.add_scalar('total_reward_per_episode', cum_rew, episode)
env.close() 