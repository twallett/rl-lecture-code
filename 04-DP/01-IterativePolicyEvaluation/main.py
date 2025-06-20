#%%
from Utils.env import GridWorldEnv
from model import IterativePolicyEvaluation

SEED = 123 
SIZE = 3
GAMMA = 0.9
EPISODES = 1000
FPS = 0

env = GridWorldEnv(render_mode="human",
                   size=SIZE)

env.metadata['render_fps'] = FPS
NUM_ACTIONS = env.action_space.n
NUM_STATES = env.observation_space.n

model = IterativePolicyEvaluation(num_states = NUM_STATES,
                                  gamma=GAMMA,
                                  seed = SEED)

for episode in range(EPISODES):
    deterministic_state = env.reset(seed=SEED)
    done = False
    while not done:
        sampled_states = []
        sampled_rewards = []
        for action in range(NUM_ACTIONS):
            state, reward, terminated = env.step(action)
            sampled_states.append(state)
            sampled_rewards.append(reward)
        env.v = model.update(deterministic_state, sampled_states, sampled_rewards)
        deterministic_state, reward, terminated = env.step(action, deterministic=True)
        env.render()
        done = terminated
env.close() 