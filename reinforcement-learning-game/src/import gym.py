import gym
from gym import spaces
import numpy as np
import pygame

# ==========================
# Custom RL Environment
# ==========================
class CoinCollectorEnv(gym.Env):
    """
    A simple reinforcement learning environment:
    - Agent moves in 2D space to collect coins
    - Reward = +1 per coin collected, -0.01 per step to encourage efficiency
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, grid_size=10, max_steps=50):
        super(CoinCollectorEnv, self).__init__()
        
        self.grid_size = grid_size
        self.max_steps = max_steps
        
        # Action space: 0=up,1=down,2=left,3=right
        self.action_space = spaces.Discrete(4)
        
        # Observation space: agent x,y and coin x,y
        self.observation_space = spaces.Box(low=0, high=grid_size-1,
                                            shape=(4,), dtype=np.int32)
        
        self.reset()
        
        # Pygame setup
        self.screen_size = 400
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        pygame.display.set_caption("RL Coin Collector")
        self.clock = pygame.time.Clock()
        self.cell_size = self.screen_size // self.grid_size

    def reset(self):
        self.agent_pos = np.array([0, 0])
        self.coin_pos = np.random.randint(0, self.grid_size, size=2)
        self.steps = 0
        return self._get_obs()
    
    def _get_obs(self):
        return np.concatenate([self.agent_pos, self.coin_pos])
    
    def step(self, action):
        # Apply action
        if action == 0 and self.agent_pos[1] > 0:  # Up
            self.agent_pos[1] -= 1
        elif action == 1 and self.agent_pos[1] < self.grid_size-1:  # Down
            self.agent_pos[1] += 1
        elif action == 2 and self.agent_pos[0] > 0:  # Left
            self.agent_pos[0] -= 1
        elif action == 3 and self.agent_pos[0] < self.grid_size-1:  # Right
            self.agent_pos[0] += 1
        
        # Reward
        reward = -0.01  # small step penalty
        done = False
        
        if np.array_equal(self.agent_pos, self.coin_pos):
            reward += 1.0
            self.coin_pos = np.random.randint(0, self.grid_size, size=2)
        
        self.steps += 1
        if self.steps >= self.max_steps:
            done = True
        
        return self._get_obs(), reward, done, {}
    
    def render(self, mode='human'):
        self.screen.fill((0, 0, 0))  # black background
        
        # Draw coin
        pygame.draw.rect(self.screen, (255, 223, 0), 
                         (self.coin_pos[0]*self.cell_size, self.coin_pos[1]*self.cell_size,
                          self.cell_size, self.cell_size))
        
        # Draw agent
        pygame.draw.rect(self.screen, (0, 128, 255),
                         (self.agent_pos[0]*self.cell_size, self.agent_pos[1]*self.cell_size,
                          self.cell_size, self.cell_size))
        
        pygame.display.flip()
        self.clock.tick(10)
    
    def close(self):
        pygame.quit()


# ==========================
# Quick Test
# ==========================
if __name__ == "__main__":
    env = CoinCollectorEnv(grid_size=8, max_steps=30)
    obs = env.reset()
    done = False
    
    while not done:
        env.render()
        action = env.action_space.sample()  # random action
        obs, reward, done, info = env.step(action)
        print(f"Obs: {obs}, Reward: {reward}")
    
    env.close()
