#%%
from Utils.env import GridWorldEnv
from model import MCPrediction

SEED = 123 
SIZE = 3
GAMMA = 0.9
EPISODES = 10
FPS = 1

env = GridWorldEnv(render_mode="human",
                   size=SIZE)

env.metadata['render_fps'] = FPS
NUM_ACTIONS = env.action_space.n
NUM_STATES = env.observation_space.n

model = MCPrediction(num_states = NUM_STATES,
                     num_actions=NUM_ACTIONS,
                     gamma=GAMMA,
                     seed = SEED)

for episode in range(EPISODES):
    state = env.reset(seed=SEED)
    done = False
    sampled_states = [state]
    sampled_rewards = []
    t = 0 
    while not done:
        action = model.pi()
        state, reward, terminated = env.step(action)
        if not terminated:
            sampled_states.append(state)
        sampled_rewards.append(reward)
        t += 1
        env.render()
        done = terminated
    env.v = model.update(sampled_states, sampled_rewards, t)
env.close() 
