#%%
import gymnasium as gym
import numpy as np

class Node:
    
    def __init__(self, state, num_actions):
        self.state = state
        self.num_actions = num_actions
        self.child_nodes = {act:None for act in range(num_actions)}
        self.q = 0
        self.n = 0
        self.depth = 0 
        self.parent = None
        
    def add_children(self, node, child_state, action):
        child_node = Node(child_state, self.num_actions)
        child_node.depth = node.depth + 1 
        child_node.parent = node
        self.child_nodes[action] = child_node
        
class MCTS:
    
    def __init__(self, env_name, simulations, c, iterations, max_depth, gamma=0.99):  # Added gamma parameter
        self.sim_env = gym.make(env_name)
        self.num_actions = self.sim_env.action_space.n
        self.simulations = simulations
        self.c = c
        self.iterations = iterations
        self.max_depth = max_depth
        self.gamma = gamma  # Discount factor for future rewards
        
    def update(self, node):
        root_node = node
        for iterations in range(self.iterations):
            action = self.select(node)
            self.expand(node, action)
            node = node.child_nodes[action]  
            total_cum_rew = 0
            for simulation in range(self.simulations):
                cum_rew = self.simulate(node)
                total_cum_rew += cum_rew
            self.backpropagate(node, (total_cum_rew/self.simulations))
            node = node.parent
        action = self.select(root_node)
        child_node_selected = root_node.child_nodes[action]
        return action, child_node_selected
    
    def return_to_root(self, node):
        while node.depth != 0:
            node = node.parent
        return node
    
    def backpropagate(self, node, avg_rew):
        discount = 1.0
        while node is not None:
            node.n += 1
            node.q += (avg_rew * discount) / node.n
            discount *= self.gamma  # Apply discount factor as we move up the tree
            node = node.parent
    
    def simulate(self, node):
        _, _ = self.sim_env.reset()
        unwrapped_env = self.sim_env.unwrapped
        unwrapped_env.state = node.state
        done = False
        cum_rew = 0
        depth = 0
        discount = 1.0  # Initialize discount factor
        
        while not done and depth < self.max_depth:
            pole_angle = unwrapped_env.state[2]
            angular_velocity = unwrapped_env.state[3]
            
            # More intelligent action selection
            if abs(pole_angle) < 0.2 and abs(angular_velocity) < 1.0:
                # If pole is near vertical and not moving too fast, choose action to stabilize
                action = 1 if pole_angle > 0 else 0
            else:
                # More aggressive correction when pole is tilting
                action = 0 if pole_angle > 0 else 1
            simulated_state, reward, terminated, truncated, info = self.sim_env.step(action)
            cum_rew += reward * discount  # Apply discount to current reward
            discount *= self.gamma  # Update discount for next step
            done = terminated or truncated
            depth += 1
        
        return cum_rew    
    
    def expand(self, node, action):
        parent_state = node.state
    
        _, _ = self.sim_env.reset()
        unwrapped_env = self.sim_env.unwrapped
        unwrapped_env.state = parent_state
        child_state, reward, terminated, truncated, info = self.sim_env.step(action)
        node.add_children(node, child_state, action)
    
    def select(self, parent_node):
        actions = []
        for action in range(self.num_actions):
            child_node = parent_node.child_nodes[action]
            if child_node == None:
                actions.append(float('inf'))
            else:
                actions.append((child_node.q / child_node.n) + self.c * np.sqrt( (np.log(parent_node.n)) / child_node.n))

        return np.argmax(actions)