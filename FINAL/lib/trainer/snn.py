import os
import sys
import time
import logging

import numpy as np

import torch
import torch.cuda as cuda
import bindsnet

import lib
from lib.utils import *
from lib.agent import load_agent, PongAgent
from lib.environment import GymEnvironment
from lib.trainer.pipeline import PongPipeline


__all__ = ['SNNTrainer']


def _init_directory(model_name):
    # information saving directory
    # save model checkpoint and episode history
    if not os.path.exists(model_name):
        os.makedirs(model_name)

    save_dir = model_name
    logging.info(
        f'All object (model checkpoint, trainning history, ...) would save in {save_dir}')
    return save_dir


class SNNTrainer():
    def __init__(self, config) -> None:
        self.cfg = load_config(config)

        self.name = self.cfg['name']
        self.save_dir = _init_directory(self.name)
        self.device = init_torch_device(self.cfg['device'])
        self.env = GymEnvironment(self.cfg['env']['env_name'])
        self.agent = self._init_agent(self.cfg['agent'], self.env, self.name)
        self.pipeline = PongPipeline(
                self.agent,
                self.env,
                output = "Output Layer",
                time = self.cfg['agent']['sim_time'],
                )

        self.recorder = Recorder(['episode', 'train_reward', 'test_reward'])
        logging.info('SNNTrainer initalization done.')

    @property
    def record_title(self):
        return self.recorder.record_column

    def save(self, path = None, episode_note = None) -> None:
        path = default_to(path, self.save_dir)
        self.agent.save(path, episode_note)
        if episode_note is None:
            save_object(os.path.join(path, 'config.pkl'), self.cfg)


    def train(self, config = None) -> None:
        config = default_to(config, self.cfg)
        episodes = config['episodes']
        checkpoint = config['checkpoint']

        logging.info('Start training ...')

        for i in range(1, episodes+1):
            start_time = time.time()

            self._episode(i, test = False)

            if i % 100 == 0:
                self._episode(i, test = True)

            if i % checkpoint == 0:
                self.save(episode_note = i-1)

            logging.info(
                f'Episode: {i} / {episodes}, takes {time.time() - start_time:.4f} seconds')

        self.save(episode_note = episodes)
        self.recorder.write(self.save_dir, config['name'])
        logging.info('Training complete.\n')

    def _episode(self, it, test = False) -> None:
        self.pipeline.network.learning = True
        reward = self.pipeline.episode(it, train = True)

        if test:
            self.pipeline.network.learning = False
            # network_temp = self.pipeline.network.clone()
            rounds = self.cfg['env']['test_num']
            test_reward = 0.

            logging.info('Start testing ...')
            for i in range(rounds):
                test_reward += self.pipeline.episode(it, train = False, test_seed = i)
                logging.info(f'Progress: {i + 1} / {rounds}')

            test_reward /= rounds

            logging.info(
                f"\nEpisode: {it} - "
                f"testing reward: {test_reward:.2f}\n"
            )
        else:
            test_reward = np.nan

        self.recorder.insert((it, reward, test_reward))


    def _init_agent(self, config, env, name, device=None):
        if config['model_type'].lower() != 'snn':
            raise RuntimeError('Only "snn" base agent can be trained in SNNTrainer.')

        if os.path.isfile(os.path.join(self.save_dir, 'agent.pkl')):
            agent = load_agent(os.path.join(self.save_dir, 'agent.pkl'),
                    os.path.join(self.save_dir, self.name + '.pt'))
        else:
            agent = PongAgent(
                    name,
                    config['model_type'],
                    config['model_config'],
                    config['preprocess'],
                    device
                    )

        return agent
