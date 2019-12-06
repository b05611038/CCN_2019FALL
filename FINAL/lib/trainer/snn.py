import os
import sys
import time

import numpy as np

import torch
import torch.cuda as cuda
import bindsnet

import lib
from lib.utils import *
from lib.agent import PongAgent
from lib.environment import GymEnvironment
from lib.trainer.pipeline import PongPipeline 


__all__ = ['SNNTrainer']


class SNNTrainer(object):
    def __init__(self, config):
        self.cfg = load_config(config)

        self.name = self.cfg['name']
        self.save_dir = self._init_directory(self.name)
        self.device = self._init_device(self.cfg['device'])
        self.env = self._init_env(self.cfg['env'])
        self.agent = self._init_agent(self.cfg['agent'], self.env)


    def save(self):
        pass

    def train(self):
        pass

    def _init_agent(self, config, env):
        # if config['model_type'].lower() != 'snn':
        #     raise RuntimeError('Please ensure that only snn base agent can be trained in SNNTrainer.')

        # agent = PongAgent

    def _init_env(self, config):
        env = GymEnvironment(config['env_name'])
        return env

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


