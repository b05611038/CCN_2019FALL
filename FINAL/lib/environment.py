import gym
import numpy as np

import torch
import bindsnet
from bindsnet.environment import GymEnvironment

from .utils import *


__all__ = ['ANNEnvironment', 'SNNEnvironment']


class ANNEnvironment(object):
    def __init__(self, env_name):
        self.env = gym.make(env_name)

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def seed(self, seed):
        '''
        Control the randomness of the environment
        '''
        self.env.seed(seed)
        return None

    def reset(self):
        observation = self.env.reset()
        return np.array(observation)

    def close(self):
        self.env.close()
        return None

    def step(self, action):
        '''
        When running pg:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)
            reward: int
                if opponent wins, reward = +1 else -1
            done: bool
                whether reach the end of the episode?
        '''
        if not self.env.action_space.contains(action):
            raise ValueError('Ivalid action!!')

        observation, reward, done, info = self.env.step(action)

        return np.array(observation).astype('uint8'), reward, done, info

    def render(self):
        return self.env.render(mode = 'rgb_array')

    def get_action_space(self):
        return self.action_space

    def get_observation_space(self):
        return self.observation_space

    def get_random_action(self):
        return self.action_space.sample()


class SNNEnvironment(GymEnvironment):
    def __init__(self, env_name):
        super(SNNEnvironment, self).__init__(env_name)

        self.observation_space = self.env.observation_space

    def seed(self, seed):
        '''
        Control the randomness of the environment
        '''
        self.env.seed(seed)

    def render(self):
        return self.env.render(mode = 'rgb_array')


