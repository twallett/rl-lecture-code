#%%

# how to construct an open ai gym environment from scratch: https://www.gymlibrary.dev/content/environment_creation/

import gymnasium as gym
import pygame
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

class GridWorldEnv(gym.Env):
    
    metadata = {"render_modes": ["human"], "render_fps": 10} # "human" render_mode means env. is designed to render in a format that a human can see real time.
    
    def __init__(self, render_mode=None, size=3):
        
        # Pygame Variables
        # ----------------------------------------------------------------
        pygame.init()
        pygame.font.init()
        self.size = size  
        self.window_size = 512
        self.render_mode = render_mode
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.window = None  
        self.clock = None
        self.font = pygame.font.Font(None, 36) 
        
        # Open AI Variables
        # ----------------------------------------------------------------
        self.observation_space = spaces.Discrete(self.size * self.size)
        self.action_space = spaces.Discrete(4)
        self.action_to_direction = {
            0: np.array([1,0]),  # down
            1: np.array([0,1]),  # right
            2: np.array([-1,0]), # up
            3: np.array([0,-1]), # left
        }
        
        # Metrics
        # ----------------------------------------------------------------
        self.v = None
        self.episode = 0
        self.steps = 0
        self.steps_per_episode = []
        self.average_steps = 0
        self.average_steps_per_episode = []
        
    def reset(self, seed=None):
        super().reset(seed=seed) 
        
        # Initialize self.v
        # ----------------------------------------------------------------
        if self.v is None:
            self.v = np.zeros((self.observation_space.n))
        
        # Return and reset state and terminal state
        # ----------------------------------------------------------------
        self.state = np.array([0,0])
        self.terminal_state = np.array([self.size - 1, self.size - 1])
        observation = self.convert_2d_to_1d(self.state)
        
        # Reset steps counter and add episode
        # ----------------------------------------------------------------
        self.steps = 0
        self.episode += 1
        
        # Return Pygame Frame
        # ----------------------------------------------------------------
        if self.render_mode == "human":
            self.render()
            
        return observation
    
    def step(self, action):
        
        self.steps += 1

        # Return Environment Reward Calculation  
        # ----------------------------------------------------------------
        if type(self.terminal_state) == int:
            self.terminal_state = self.convert_1d_to_2d(self.terminal_state)
        if type(self.state) == int:
            self.state = self.convert_1d_to_2d(self.state)
        
        directions = [
            ([1, 0], self.state[0] + 1 < self.size),   # down
            ([0, 1], self.state[1] + 1 < self.size),   # right
            ([-1, 0], self.state[0] - 1 >= 0),         # up
            ([0, -1], self.state[1] - 1 >= 0)          # left
        ]
        
        values = [
            1 if np.array_equal([self.state[0] + d[0], self.state[1] + d[1]], self.terminal_state) 
            else 0 if valid 
            else -1 
            for d, valid in directions
        ]
        reward = values[action]
        step = directions[action][0]
        self.state = ([self.state[0] + step[0], self.state[1] + step[1]] if values[action] != -1 else self.state)
        terminated = np.array_equal(self.state, self.terminal_state)
        observation = self.convert_2d_to_1d(self.state)
    
        # Metrics
        # ----------------------------------------------------------------
        if terminated:
            self.steps_per_episode.append(self.steps)
            self.average = np.mean(self.steps_per_episode)
            self.average_steps_per_episode.append(self.average) 
        
        # Render Pygame Frame
        # ----------------------------------------------------------------
        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated

    def render(self):
        
        # Initialize self.v
        # ----------------------------------------------------------------
        self.v = np.array(self.v).reshape(self.size, self.size)
        if type(self.terminal_state) == int:
            self.terminal_state = self.convert_1d_to_2d(self.terminal_state)
        if type(self.state) == int:
            self.state = self.convert_1d_to_2d(self.state)
        
        # Initializing Pygame Window (Frame) and Clock (FPS)
        # ----------------------------------------------------------------
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size + 500, self.window_size + 70))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # Pygame Frame 
        # ----------------------------------------------------------------    
        frame = pygame.Surface((self.window_size + 500, self.window_size + 70))
        frame.fill((255, 255, 255))
        pix_square_size = self.window_size / self.size 

        # Draw Terminal State
        # ---------------------------------------------------------------- 
        pygame.draw.rect(
            frame,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self.terminal_state,
                (pix_square_size, pix_square_size),
            ),
        )
        
        # Draw Agent
        # ---------------------------------------------------------------- 
        pygame.draw.circle(
            frame,
            (0, 0, 0), 
            (int(self.state[1] * pix_square_size + 0.5 * pix_square_size), 
            int(self.state[0] * pix_square_size + 0.5 * pix_square_size)),
            int(pix_square_size / 3) + 3, 
        )
        pygame.draw.circle(
            frame,
            (0, 0, 255),  
            (int(self.state[1] * pix_square_size + 0.5 * pix_square_size), 
            int(self.state[0] * pix_square_size + 0.5 * pix_square_size)),
            pix_square_size / 3,
        )

        # Draw Policy 
        # ---------------------------------------------------------------- 
        arrow_surface = pygame.Surface((self.window_size, self.window_size), pygame.SRCALPHA)
        arrow_surface.set_alpha(128) 
        for row in range(self.size):
            for col in range(self.size):
                if np.array_equal([row, col], self.terminal_state):
                    continue 

                center_x = int(col * pix_square_size + 0.5 * pix_square_size)
                center_y = int(row * pix_square_size + 0.5 * pix_square_size)
                
                if self.episode == 1:
                    values = [0 for d in self.action_to_direction.values()]
                else: 
                    values = [
                            1.0 if np.array_equal([row + d[0], col + d[1]], self.terminal_state)
                            else self.v[row + d[0], col + d[1]] if 0 <= row + d[0] < self.size and 0 <= col + d[1] < self.size
                            else float('-inf') 
                            for d in self.action_to_direction.values()
                        ]
                max_val = np.max(values)
                directions = [i for i, val in enumerate(values) if val == max_val]

                for direction in directions:
                    if direction == 0:  # Down
                        start_pos = (center_x, center_y - 20)
                        end_pos = (center_x, center_y + 20)
                        arrow_tip = [(center_x - 10, center_y + 20), (center_x + 10, center_y + 20), (center_x, center_y + 30)]
                    elif direction == 1:  # Right
                        start_pos = (center_x - 20, center_y)
                        end_pos = (center_x + 20, center_y)
                        arrow_tip = [(center_x + 20, center_y - 10), (center_x + 20, center_y + 10), (center_x + 30, center_y)]
                    elif direction == 2:  # Up
                        start_pos = (center_x, center_y + 20)
                        end_pos = (center_x, center_y - 20)
                        arrow_tip = [(center_x - 10, center_y - 20), (center_x + 10, center_y - 20), (center_x, center_y - 30)]
                    elif direction == 3:  # Left
                        start_pos = (center_x + 20, center_y)
                        end_pos = (center_x - 20, center_y)
                        arrow_tip = [(center_x - 20, center_y - 10), (center_x - 20, center_y + 10), (center_x - 30, center_y)]
                    
                    if not (0 <= start_pos[0] < self.window_size and 0 <= start_pos[1] < self.window_size):
                        continue
                    if not (0 <= end_pos[0] < self.window_size and 0 <= end_pos[1] < self.window_size):
                        end_pos = (min(max(end_pos[0], 0), self.window_size), min(max(end_pos[1], 0), self.window_size))

                    pygame.draw.line(arrow_surface, (0, 0, 0), start_pos, end_pos, 6)
                    pygame.draw.polygon(arrow_surface, (0, 0, 0), arrow_tip)
                    
                    # Draw values v or action values q 
                    # ---------------------------------------------------------------- 
                    value_font_size = 24  
                    value_font = pygame.font.Font(None, value_font_size)
                    value_text = f"{self.v[row, col]:.2f}"
                    text_surface = value_font.render(value_text, False, (0, 0, 0)) 
                    text_x = center_x - text_surface.get_width() / 2
                    text_y = int((row + 1) * pix_square_size) - text_surface.get_height() - 5 
                    frame.blit(text_surface, (text_x, text_y))

        frame.blit(arrow_surface, (0, 0))
        
        # Draw Gridlines 
        # ---------------------------------------------------------------- 
        for x in range(self.size + 1):
            pygame.draw.line(
                frame,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=2,
            )
            pygame.draw.line(
                frame,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=2,
            )
        
        # Draw Metrics
        # ---------------------------------------------------------------- 
        episode_text = self.font.render(f'Episode: {self.episode}', True, 0)
        step_text = self.font.render(f'Steps: {self.steps}', True, 0)
        font_size = 28
        font = pygame.font.Font(None, font_size)
        title = font.render(f'Monte Carlo Prediction Vπ(s)', True, 0)

        text_x = 10
        text_y = self.window_size + 10
        step_text_y = self.window_size + 40

        frame.blit(episode_text, (text_x, text_y))
        frame.blit(step_text, (text_x, step_text_y))
        frame.blit(title, (text_x + 150, step_text_y - 20))
        
        # Draw Metric Plot
        # ----------------------------------------------------------------    
        episodes = np.arange(1, len(self.average_steps_per_episode) + 1)
        plt.figure(figsize=(5, 5))
        plt.plot(episodes, self.average_steps_per_episode, marker='o', color='orange')
        plt.xlabel('Episode')
        plt.ylabel('Average Steps')
        plt.title('Average Steps per Episode')
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        graph_surface = pygame.image.load(buf)
        buf.close()
        plt.close()
        graph_x = self.window_size + 20
        frame.blit(graph_surface, (graph_x, 20))
                
        if self.render_mode == "human":
            self.window.blit(frame, frame.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def convert_1d_to_2d(self, indx):
        return np.array([indx // self.size, indx % self.size])
    
    def convert_2d_to_1d(self, array):
        return (array[0] * self.size) + array[1]
        
        

        
        