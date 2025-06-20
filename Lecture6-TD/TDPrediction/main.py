#%%
from Utils.env import GridWorldEnv
from model import TDPrediction

SEED = 123 
SIZE = 3
GAMMA = 0.9
ALPHA = 0.1
EPISODES = 10
FPS = 0

env = GridWorldEnv(render_mode="human",
                   size=SIZE)

env.metadata['render_fps'] = FPS
NUM_ACTIONS = env.action_space.n
NUM_STATES = env.observation_space.n

model = TDPrediction(num_states = NUM_STATES,
                     num_actions=NUM_ACTIONS,
                     alpha=ALPHA,
                     gamma=GAMMA,
                     seed = SEED)

for episode in range(EPISODES):
    state = env.reset(seed=SEED)
    done = False
    while not done:
        action = model.pi()
        next_state, reward, terminated = env.step(action)
        env.v = model.update(state, next_state, reward)
        env.render()
        state = next_state
        done = terminated
env.close() 