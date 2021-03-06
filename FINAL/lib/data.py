import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import lib
from lib.utils import *


__all__ = ['QReplayBuffer', 'PGReplayBuffer', 'SNNReplayBuffer', 'EpisodeSet', 'SNNEpisodeSet']


class QReplayBuffer(Dataset):
    # for DQN base trainer
    def __init__(self, env, maximum, preprocess_dict, gamma = 0.99):
        #only note the game environment
        #not the enviroment object
        self.env = env
        self.maximum = maximum
        self.preprocess_dict = preprocess_dict
        self.gamma = gamma
        self.eps = 10e-7

        self.data = []
        self.rewards = []

    def reset_maximum(self, new):
        self.maximum = new
        return None

    def insert(self, observation, next_observation, action):
        if len(self.data) > self.maximum:
            self.data = self.data[1: ]
            self.rewards = self.rewards[1: ]

        self.data.append([observation, next_observation, action])

        return None

    def insert_reward(self, reward, times):
        if self.preprocess_dict['reward_decay']:
            decay_reward = (reward * (np.power(self.gamma, np.flip(np.arange(times))) / \
                    np.sum(np.power(self.gamma, np.flip(np.arange(times)))))).tolist()
            self.rewards[len(self.rewards): ] = decay_reward
        else:
            normal_reward = (reward * np.repeat(1.0, times) / np.sum(np.repeat(1.0, times))).tolist()
            self.rewards[len(self.rewards): ] = normal_reward

        return None

    def trainable(self):
        #check the buffer is ready for training
        return True if len(self.rewards) > (self.maximum // 4) else False

    def __getitem__(self, index):
        select = random.randint(0, len(self.data) - 1)
        return self.data[select][0].squeeze(0).float().detach(), self.data[select][1].squeeze(0).float().detach(), \
                torch.tensor(self.data[select][2]).long(), torch.tensor(self.rewards[select]).float()

    def __len__(self):
        return self.maximum // 8


class PGReplayBuffer(Dataset):
    # for policy gradient, actor-critic base trainer
    def __init__(self, env, maximum, preprocess_dict, gamma = 0.99):
        #only note the game environment
        #not the enviroment object
        self.env = env
        self.maximum = maximum
        self.preprocess_dict = preprocess_dict
        self.length = 0
        self.gamma = gamma
        self.eps = 10e-7

        #one elemnet in datalist is a training pair with three elements: observation, reward, action
        #the pair relationship -> model(observation) ==> action ==> reward
        self.data = []
        self.rewards = []
        self.__insert_lock = []

    def reset_maximum(self, new):
        self.maximum = new
        return None

    def new_episode(self):
        if len(self.rewards) > self.maximum:
            self.data = self.data[1: ]
            self.rewards = self.rewards[1: ]
            self.__insert_lock = self.__insert_lock[1: ]

        self.data.append([])
        self.rewards.append([])
        self.__insert_lock.append(False)
        return None

    def insert(self, observation, action):
        if self.__insert_lock[-1] != True:
            #not lock can append
            self.data[-1].append([observation.squeeze(), action])
        else:
            raise RuntimeError('Please use new_episode() before insert new episode information.')

        return None

    def insert_reward(self, reward, times, done):
        if self.__insert_lock[-1] != True:
            for i in range(times):
                if self.preprocess_dict['reward_decay']:
                    decay_reward = reward * math.pow((self.gamma), (times - 1 - i))
                    self.rewards[-1].append(decay_reward)
                else:
                    self.rewards[-1].append(reward)

        else:
            raise RuntimeError('Please use new_episode() before insert new episode information.')

        if done:
            self.__insert_lock[-1] = True

        return None

    def trainable(self):
        #check the buffer is ready for training
        return True if len(self.rewards) >= self.maximum else False

    def make(self, episode_size):
        self.observation = None
        self.action = None
        self.reward = None
        for i in range(episode_size):
            select = random.randint(0, self.maximum - 1)
            dataset = EpisodeSet(self.data[select], self.rewards[select])
            dataloader = DataLoader(dataset, batch_size = len(self.data[select]), shuffle = False)
            for iter, (obs, act, rew) in enumerate(dataloader):
                if self.observation is None:
                    self.observation = obs.squeeze()
                else:
                    self.observation = torch.cat((self.observation, obs.squeeze()), dim = 0)

                if self.action is None:
                    self.action = act.squeeze()
                else:
                    self.action = torch.cat((self.action, act.squeeze()), dim = 0)

                if self.reward is None:
                    self.reward = rew
                else:
                    self.reward = torch.cat((self.reward, rew), dim = 0)

        if self.preprocess_dict['reward_normalize']:
            mean = torch.mean(self.reward, dim = 0)
            std = torch.std(self.reward, dim = 0)
            self.reward = (self.reward - mean) / (std + self.eps)

        self.length = self.reward.size(0)
        return None

    def __getitem__(self, index):
        return self.observation[index].detach(), self.action[index].detach(), self.reward[index].detach()

    def __len__(self):
        return self.length


class EpisodeSet(Dataset):
    def __init__(self, data, rewards):
        self.data = data
        self.rewards = rewards

    def __getitem__(self, index):
        #return observation, action
        reward = torch.tensor(self.rewards[index]).float()
        return self.data[index][0].float(), self.data[index][1].float(), reward

    def __len__(self):
        return len(self.data)


class SNNReplayBuffer(Dataset):
    # for policy gradient, actor-critic base trainer
    def __init__(self, env, maximum, preprocess_dict, gamma = 0.99):
        #only note the game environment
        #not the enviroment object
        self.env = env
        self.maximum = maximum
        self.preprocess_dict = preprocess_dict
        self.length = 0
        self.gamma = gamma
        self.eps = 10e-7

        #one elemnet in datalist is a training pair with three elements: observation, reward, action
        #the pair relationship -> model(observation) ==> action ==> reward
        self.data = []
        self.rewards = []
        self.__insert_lock = []

    def reset_maximum(self, new):
        self.maximum = new
        return None

    def new_episode(self):
        if len(self.rewards) > self.maximum:
            self.data = self.data[1: ]
            self.rewards = self.rewards[1: ]
            self.__insert_lock = self.__insert_lock[1: ]

        self.data.append([])
        self.rewards.append([])
        self.__insert_lock.append(False)
        return None

    def insert(self, observation):
        if self.__insert_lock[-1] != True:
            #not lock can append
            self.data[-1].append([observation.squeeze()])
        else:
            raise RuntimeError('Please use new_episode() before insert new episode information.')

        return None

    def insert_reward(self, reward, times, done):
        if self.__insert_lock[-1] != True:
            for i in range(times):
                if self.preprocess_dict['reward_decay']:
                    decay_reward = reward * math.pow((self.gamma), (times - 1 - i))
                    self.rewards[-1].append(decay_reward)
                else:
                    self.rewards[-1].append(reward)

        else:
            raise RuntimeError('Please use new_episode() before insert new episode information.')

        if done:
            self.__insert_lock[-1] = True

        return None

    def make(self, episode_size):
        self.observation = None
        self.reward = None
        for i in range(episode_size):
            select = random.randint(0, self.maximum - 1)
            dataset = SNNEpisodeSet(self.data[select], self.rewards[select])
            dataloader = DataLoader(dataset, batch_size = len(self.data[select]), shuffle = False)
            for iter, (obs, rew) in enumerate(dataloader):
                if self.observation is None:
                    self.observation = obs.squeeze()
                else:
                    self.observation = torch.cat((self.observation, obs.squeeze()), dim = 0)

                if self.reward is None:
                    self.reward = rew
                else:
                    self.reward = torch.cat((self.reward, rew), dim = 0)

        if self.preprocess_dict['reward_normalize']:
            mean = torch.mean(self.reward, dim = 0)
            std = torch.std(self.reward, dim = 0)
            self.reward = (self.reward - mean) / (std + self.eps)

        self.length = self.reward.size(0)
        return None

    def __getitem__(self, index):
        return self.observation[index].detach(), self.reward[index].detach()

    def __len__(self):
        return self.length


class SNNEpisodeSet(Dataset):
    def __init__(self, data, rewards):
        self.data = data
        self.rewards = rewards

    def __getitem__(self, index):
        #return observation, action
        reward = torch.tensor(self.rewards[index]).float()
        return self.data[index][0].float(), reward

    def __len__(self):
        return len(self.data)


