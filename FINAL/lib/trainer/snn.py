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
        self.agent = self._init_agent(self.cfg['agent'], self.env, self.name)
        self.pipeline = PongPipeline(
                self.agent,
                self.env,
                output = "Output Layer",
                time = self.cfg['agent']['sim_time'],
                )

        self.record_title = ['episode', 'train_reward', 'test_reward']
        self.recorder = Recorder(self.record_title)
        print('SNNTrainer inital done.')

    def save(self, path = None, episode_note = None):
        if path is None:
            path = self.save_dir

        self.agent.save(path, note) 
        return None

    def train(self, config = None):
        if config is None:
            cfg = self.cfg

        episodes = cfg['episodes']
        checkpoint = cfg['checkpoint']
        print('Start training ...')
        for i in range(episodes):
            start_time = time.time()
            if i % 100 == 0 and i != 0:
                self._episode(i, test = True)
            else:
                self._episode(i, test = False)

            if i % cfg['checkpoint'] == 0 and i != 0:
                self.save()

            print('Episode: %d / %d, cause %.4f seconds' % (i + 1, episodes, time.time() - start_time))

        self.recorder.write(self.save_dir, cfg['name'])
        print("Training complete.\n")
        return None

    def _episode(self, iter, test = False):
        if test:
            self.pipeline.network.learning = False
        else:
            self.pipeline.network.learning = True

        reward = self.pipeline.episode(iter, train = True)
        if test:
            rounds = self.config['env']['test_num']
            test_reward = 0.
            for i in range(rounds):
                test_reward += self.pipeline.episode(iter, train = False, test_seed = i)

            test_reward /= rounds
            print(
                f"\nEpisode: {iter} - "
                f"testing reward: {test_reward:.2f}\n"
            )

        if test:
            self.recorder.insert((iter, reward, test_reward))
        else:
            self.recorder.insert((iter, reward, np.nan))

        return None

    def _init_agent(self, config, env, name):
        if config['model_type'].lower() != 'snn':
            raise RuntimeError('Please ensure that only snn base agent can be trained in SNNTrainer.')

        agent = PongAgent(
                name, 
                config['model_type'],
                config['model_config'],
                config['preprocess'],
                self.device
                )

        return agent

    def _init_env(self, env_cfg):
        env = GymEnvironment(env_cfg['env_name'])
        
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


