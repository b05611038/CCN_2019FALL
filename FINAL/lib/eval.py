import os
import sys
import time

import numpy as np

import torch
import bindsnet

import lib
from lib.utils import *
from lib.agent import load_agent, PongAgent
from lib.environment import GymEnvironment
from lib.plot import TrainingPlot, VideoMaker


__all__ = ['AgentPlayer']


class AgentPlayer(object):
    def __init__(self, agent_config_path, agent_model_path, env = 'Pong-v0', device = None):
        self.device = init_torch_device(device)

        self.agent = load_agent(agent_config_path, agent_model_path, device = self.device)
        self.save_dir = self._init_directory(self.agent.name)
        self.models = self._get_model_checkpoint(self.agent.name)
        self.env = GymEnvironment(env)
        self.ploter = TrainingPlot(self.agent.name)

    def demo(self, sample_times):
        print('Plot training history ...')
        self.ploter.plot_all()

        print('Start make checkpoint models interact with environmnet ...')
        maker = VideoMaker(self.agent.name)

        for iter in range(len(self.models)):
            self.agent.load_model(self.models[iter])
            scores, videos = self._play_game(sample_times)
            index = self._max_score(scores)
            maker.insert_video(np.asarray(videos[index]))
            maker.make(self.save_dir, self.models[iter].split('/')[-1].replace('.pt', ''))

            print('Progress:', iter + 1, '/', len(self.models))

        print('All video saving done.')
        return None

    def _play_game(self, times):
        scores = []
        videos = []
        for i in range(times):
            observation = self.env.reset()
            videos.append([])
            self.agent.insert_memory(observation)
            scores.append(0.0)
            done = False
            while not done:
                action, _processed, _model_out = self.agent.make_action(observation)
                observation_next, reward, done, _ = self.env.step(action)
                scores[i] += reward
                videos[i].append(observation)
                observation = observation_next

        return scores, np.asarray(videos)

    def _max_score(self, scores):
        max_score = -1
        record = -1
        for i in range(len(scores)):
            if scores[i] > max_score:
                record = i
                max_score = scores[i]

        return record

    def _get_model_checkpoint(self, path):
        if os.path.isdir(path):
            files = os.listdir(path)
            model_list = [os.path.join(path, file)
                          for file in os.listdir(path)
                          if file.endswith('.pt')]
            return model_list

        else:
            print('There is not any checkpoint can used for showing.')
            exit(0)

    def _init_directory(self, model_name):
        # information saving directory
        # save model checkpoint and episode history
        if not os.path.exists(model_name):
            os.makedirs(model_name)

        save_dir = model_name
        print('All object (model checkpoint, trainning history, ...) would save in', save_dir)
        return save_dir
