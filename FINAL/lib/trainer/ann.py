import os
import sys
import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam, RMSprop
from torch.utils.data import DataLoader

import lib
from lib.utils import *
from lib.data import QReplayBuffer, PGReplayBuffer
from lib.agent import PongAgent
from lib.environment import GymEnvironment


__all__ = ['ANNTrainer']


class ANNTrainer(object):
    def __init__(self, config):
        self.cfg = load_config(config)
        self.model_name = self.cfg['model_name']
        self.save_dir = self._init_directory(self.model_name)
        self.device = self._init_device(self.cfg['device'])
        self.policy = self._check_policy(self.cfg['policy'])
        self.env = self._init_env(self.cfg['env'])
        self.agent = self._init_agent(self.cfg['agent'], self.env, self.name)
        self.agent.set_policy(self.policy)
        self.model = self.agent.model
        self.optim = self._init_optimizer(self.model, self.cfg['optim'])

        self.eps = 10e-7

        self.reward_preprocess = self.cfg['dataset']['reward_preprocess']
        if self.policy == 'DDQN':
            self.dataset = QReplayBuffer(env = env, maximum = self.cfg['dataset']['maximum'],
                    preprocess_dict = reward_preprocess)
        else:
            self.dataset = PGReplayBuffer(env = env, maximum = self.cfg['dataset']['maximum'],
                    preprocess_dict = reward_preprocess)

        self.recorder = Recorder(['episode', 'loss', 'train_reward', 'test_reward'])

    def save(self, path = None, episode_note = None):
        if path is None:
            path = self.save_dir

        self.agent.save(path, episode_note)
        if episode_note is None:
            save_object(os.path.join(path, 'config.pkl'), self.cfg)

        return None

    def play(self):
        raise NotImplementedError()

    def _init_agent(self, config, env, name):
        if config['model_type'].lower() != 'snn':
            raise RuntimeError('Only "snn" base agent can be trained in SNNTrainer.')

        agent = PongAgent(
                name,
                config['model_type'],
                config['model_config'],
                config['preprocess'],
                self.device
                )

        return agent

    def _init_optimizer(self, model, cfg):
        if cfg['name'].lower() == 'sgd':
            return SGD(model.parameters(), **cfg['args'])
        elif cfg['name'].lower() == 'rmsprop':
            return RMSprop(model.parameters(), **cfg['args'])
        elif cfg['name'].lower() == 'adam':
            return Adam(model.parameters(), **cfg['args'])
        else:
            raise ValueError('Optimizer:', cfg['name'], 'is not implemented in ANNBaseTrainer.')

    def _init_env(self, env_cfg):
        env = GymEnvironment(env_cfg['env_name'])
        return env

    def _check_policy(self, policy):
        policy_list = ['PO', 'PPO', 'DDQN', 'A2C']
        if policy not in policy_list:
            raise ValueError(policy, 'is not in ' + str(policy_list))

        return policy

    def _init_device(self, device):
        device = init_torch_device(device)
        return device

    def _init_directory(self, model_name):
        # information saving directory
        # save model checkpoint and episode history
        if not os.path.exists(model_name):
            os.makedirs(model_name)

        save_dir = model_name
        print('All object (model checkpoint, trainning history, ...) would save in', save_dir)
        return save_dir


