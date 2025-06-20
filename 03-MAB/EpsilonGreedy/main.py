#%%
from model import EpsilonGreedy
from Utils.env import *
from Utils.plot import *
import warnings

warnings.filterwarnings("ignore")

environment = "bernoulli"
n_arms = 2
arm_means = [[0.9, 0.1],[0.1, 0.9]]
obs = 500
recommendations = 10000
random_seed = 123
epsilon = 0.3

data = create_environment(env = environment,
                          n_arms = n_arms,
                          arm_means = arm_means, 
                          observations = obs, 
                          random_seed = random_seed)

model = EpsilonGreedy(n_arms = n_arms, epsilon = epsilon, random_seed=random_seed)

table, rewards, matrix = model.train(data)

# violinplot_environment(data, arm_means)

data_average_plot(data, arm_means)

data_cumulative_plot(data, arm_means)

model_average_plot(data, rewards, matrix, arm_means)

model_cumulative_plot(data, rewards, matrix, arm_means)

param = model.save()
# %%
