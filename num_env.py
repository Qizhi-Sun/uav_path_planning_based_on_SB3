import gymnasium as gym
import numpy as np
import math

x_goal = 13
y_goal = 34
z_goal = 7
class GoLeftEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.state = np.array([2,2,2,11,32,5])
        self.action_space = gym.spaces.Box(low=np.array([-0.6, -0.6, -0.6]), high=np.array([0.6, 0.6, 0.6]), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=np.array([0,0,0,-50,-50,-10]), high=np.array([50,50,10,50,50,10]), dtype=np.float32)
        self.time = 0
    def reset(self, seed = None):
        self.state = np.array([2,2,2,11,32,5], dtype=np.float32)
        self.time = 0
        return self.state, {}


    def step(self, action):
        self.state[:3] += action[:3]
        if self.state[0]<1 or self.state[0]>49 or self.state[1]<1 or self.state[1]>49 or self.state[2]<1 or self.state[2]>9:
            r_edge = -5
        elif self.state[0]<=0 or self.state[0]>=50 or self.state[1]<=0 or self.state[1]>=50 or self.state[2]<=0 or self.state[2]>=10:
            r_edge = -10
        else:
            r_edge = 0

        self.state[:2] = np.clip(self.state[:2], 0, 50)
        self.state[2] = np.clip(self.state[2], 0, 10)

        x_diff = self.state[0] - x_goal
        y_diff = self.state[1] - y_goal
        z_diff = self.state[2] - z_goal
        distance_to_goal = math.hypot( x_diff, y_diff, z_diff)
        self.state[3] = x_diff
        self.state[4] = y_diff
        self.state[5] = z_diff
        r = r_edge + distance_to_goal * -0.05
        done = distance_to_goal <= 2
        reward = 15.0 if done else r
        self.time += 1
        truncated = 1 if self.time >=200 else 0
        if done:
            print("!!!!!!!!!!!!!!! UAV HAVE BEEN FINAL !!!!!!!!!!!!!!!")
        return np.array(self.state, dtype=np.float32), float(reward), bool(done), bool(truncated), {}

    def render(self, mode='console'):
        print(self.state)


