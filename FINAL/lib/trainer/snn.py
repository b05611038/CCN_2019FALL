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
        print('SNNTrainer initalization done.')

    def save(self, path = None, episode_note = None):
        if path is None:
            path = self.save_dir

        self.agent.save(path, episode_note)
        if episode_note is None:
            save_object(os.path.join(path, 'config.pkl'), self.cfg)

        return None

    def train(self, config = None):
        if config is None:
            cfg = self.cfg

        episodes = cfg['episodes']
        checkpoint = cfg['checkpoint']

        print('Start training ...')
        for i in range(episodes):
            start_time = time.time()

            self._episode(i + 1, test = False)

            if i % 100 == 0 and i != 0:
                self._episode(i + 1, test = True)

            if i % checkpoint == 0 and i != 0:
                self.save(episode_note = i)

            print(f'Episode: {i + 1} / {episodes}, takes {time.time() - start_time:.4f} seconds')

        self.save(episode_note = episodes)
        self.recorder.write(self.save_dir, cfg['name'])
        print("Training complete.\n")
        return None

    def _episode(self, iter, test = False):
        self.pipeline.network.learning = True
        reward = self.pipeline.episode(iter, train = True)

        if test:
            self.pipeline.network.learning = False
            network_temp = self.pipeline.network.clone()
            rounds = self.cfg['env']['test_num']
            test_reward = 0.
            print('Start testing ...')
            for i in range(rounds):
                test_reward += self.pipeline.episode(iter, train = False, test_seed = i)
                print('Progress: %d / %d' % (i + 1, rounds))

            test_reward /= rounds
            print(
                f"\nEpisode: {iter} - "
                f"testing reward: {test_reward:.2f}\n"
            )
        else:
            test_reward = np.nan

        self.recorder.insert((iter, reward, test_reward))
        return None

    def _init_agent(self, config, env, name):
        if config['model_type'].lower() != 'snn':
            raise RuntimeError('Only "snn" base agent can be trained in SNNTrainer.')

        if os.path.isfile(os.path.join(self.save_dir, 'agent.pkl')):
            agent = load_agent(os.path.join(self.save_dir, 'agent.pkl'),
                    os.path.join(self.save_dir, self.model_name + '.pt'))
        else:
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


