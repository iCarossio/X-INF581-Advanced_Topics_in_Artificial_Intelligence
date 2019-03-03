import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.spaces.multi_discrete import MultiDiscrete
import numpy as np

class MyEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    action_space = MultiDiscrete([3,2,4])
    state_space = MultiDiscrete([3,2,4])

    def __init__(self):
        pass

    def step(self, action):
        return self.state_space.sample(), 0, False, None

    def reset(self):
        return self.state_space.sample()

    def render(self): 
        pass
