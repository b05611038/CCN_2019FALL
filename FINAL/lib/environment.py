import gym
import numpy as np

import torch
import bindsnet
from bindsnet.environment import Environment


__all__ = ['GymEnvironment']


class GymEnvironment(Environment):
    def __init__(self, env_name = 'Pong-v0', train = True, **kwargs):
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.train = train
        if not self.train:
            # fix the testing envirnment
            self.seed(0)

        if env_name == 'Pong-v0':
            self.action_space = gym.spaces.discrete.Discrete(n = 3)
        else:
            self.action_space = self.env.action_space

        self.observation_space = self.env.observation_space

        self.clip_rewards = kwargs.get("clip_rewards", False)

    def seed(self, seed):
        '''
        Control the randomness of the environment
        '''
        self.env.seed(seed)
        return None

    def reset(self):
        self.history = {'observations': [], 'actions': [], 'rewards': []}
        observation = self.env.reset()
        self.history['observations'].append(observation)
        return np.array(observation)

    def close(self):
        self.env.close()
        return None

    def preprocess(self):
        return None

    def step(self, action, tensor = False):
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

        self.history['actions'].append(action)

        observation, reward, done, info = self.env.step(action)
        observation = np.array(observation)
        self.history['observations'].append(observation)

        if self.clip_rewards:
            reward = np.sign(reward)

        self.history['rewards'].append(reward)
        if tensor:
            observation = torch.tensor(observation)

        return observation, reward, done, info

    def render(self):
        return self.env.render(mode = 'rgb_array')

    def get_history(self):
        return self.history

    def get_action_space(self):
        return self.action_space

    def get_observation_space(self):
        return self.observation_space

    def get_random_action(self):
        return self.action_space.sample()


