import os
import random
import numpy as np

import torch
import torch.cuda as cuda
from torch.distributions import Categorical

import lib
from lib.utils import *
from lib.agent.base import Agent
from lib.agent.preprocess import Transform
from lib.agent.ann import ANN
from lib.agent.snn import SNN


__all__ = ['PongAgent', 'load_agent']


def load_agent(path):
    state = load_pickle_obj(path)
    if state['type'] != 'PongAgent':
        raise TypeError(path, ' is not a file of Agent object')

    agent = PongAgent(state['args'])
    agent.model = state['model']
    agent.model = agent.model.to(agent.device)
    return agnet


class PongAgent(Agent):
    def __init__(self, name, model_type, model_config, preprocess_config, device, **kwargs):
        super(PongAgent, self).__init__(name, 'Pong-v0')

        self.device = init_torch_device(device)

        self.valid_action = kwargs.get('valid_action', None)
        self.action = self._init_action(self.valid_action)

        self.model = self._init_model(model_type, model_config)
        self.transform = Transform(preprocess_config, self.device)

        self.model_type = model_type
        self.model_config = model_config
        self.preprocess_config = preprocess_config
        self.memory = None
        
    def save(self, directory, note = None):
        state = {'type': 'PongAgent'}

        state['args'] = {
                'name': self.name,
                'model_type': self.model_type,
                'model_config': self.model_config,
                'preprocess_config': self.preprocess_config,
                }

        state['model'] = self.model
        if note is None:
            save_object(os.path.join(directory, self.name), state)
        else:
            save_object(os.path.join(directory, self.name + note), state)

        return None

    def load(self, path):
        state = load_pickle_obj(path)
        if state['type'] != 'PongAgent':
            raise TypeError(path, ' is not a file of Agent object')

        self.model_type = state['args']['model_type']
        self.model_config = state['args']['model_config']
        self.transform = Transform(state['args']['preprocess_config'], self.device)
        self.model = state['model']
        return None

    def rename(self, new_name):
        self.name = new_name
        return None

    def eval(self):
        self.model = self.model.eval()
        return None

    def train(self):
        self.model = self.model.train()
        return None

    def make_action(self):
        #return processed model observation and action
        if self.observation_preprocess['minus_observation'] == True:
            if self.memory is None:
                raise RuntimeError('Please insert init memory before playing a game.')

        self.model = self.model.eval()
        processed = self.preprocess(observation)
        processed = processed.to(self.device)
        input_processed = processed.unsqueeze(0)
        output, _ = self.model(input_processed)
        self.insert_memory(observation)
        action = self._decode_model_output(output)
        return action, processed.cpu().detach(), output.cpu().detach()

    def random_action(self):
        return self.valid_action[random.randint(0, len(self.valid_action) - 1)]

    def insert_memory(self, observation):
        observation = self.preprocess(observation, mode = 'init')
        self.memory = observation.to(self.device)
        return None

    def preprocess(self, observation, mode = 'normal'):
        if mode == 'normal':
            return self.transform(observation, self.memory)
        elif mode == 'init':
            return self.transform.insert_init_memory(observation)

    def _decode_model_output(self, output, mode = 'sample'):
        if mode == 'argmax':
            _, action = torch.max(output, 1)
            action_index = action.cpu().detach().numpy()[0]
            action = self.valid_action[action_index]
            return action
        elif mode == 'sample':
            try:
                output = output.detach().squeeze().cpu()
                m = Categorical(output)
                action_index = m.sample().numpy()
                action = self.valid_action[action_index]
                return action
            except RuntimeError:
                #one numbers in  probability distribution is zero
                _, action = torch.max(output, 0)
                action_index = action.cpu().detach().numpy()[0]
                action = self.valid_action[action_index]
                return action

    def _check_memory(self):
        if len(self.memory) > self.max_memory_size:
            self.memory = self.memory[-self.max_memory_size: ]
        return None

    def _init_action(self, select):
        if isinstance(select, (list, tuple)):
            return select
        elif select is None:
            return (2, 3, 4)
        else:
            raise TypeError('Argument: valid_action must be a list or typle.')

    def _init_model(self, model_type, model_config):
        if not isinstance(model_config, dict):
            raise TypeError('Type of argument: model_config must be a dict object.')

        if model_type.lower() == 'ann':
            model = ANN(num_actions = len(self.action), **model_config)
        elif model_type.lower() == 'snn':
            model = SNN(num_actions = len(self.action), **model_config)
        else:
            raise ValueError(model_type, ' not a valid selection for choosing agent model.')

        return model


