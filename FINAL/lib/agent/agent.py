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


def load_agent(agent_cfg, model_file, note = None):
    if note is None:
        state = load_pickle_obj(agent_cfg)
        if state['type'] != 'PongAgent':
            raise TypeError(path, ' is not a file of Agent object')

        agent = PongAgent(state['args'])
        if agent.model_type.lower() == 'snn':
            agent.model = bindsnet.network.network.load(model_file)
        elif agent.model_type.lower() == 'ann':
            agent.model = agent.model.cpu()
            agent.model.load_state_dict(torch.load(model_file))

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
        self.policy = None
        self.memory = None
        
    def save(self, directory, note = None):
        state = {'type': 'PongAgent'}

        state['args'] = {
                'name': self.name,
                'model_type': self.model_type,
                'model_config': self.model_config,
                'preprocess_config': self.preprocess_config,
                'policy': self.policy,
                }

        save_object(os.path.join(directory, 'agent'), state)
        self.model = self.model.cpu()
        if note is None:
            if self.model_type.lower() == 'snn':
                self.model.save(os.path.join(directory, self.name + '.pt'))
            elif self.model_type.lower() == 'ann':
                torch.save(self.model.state_dict(), os.path.join(directory, self.name + '.pt'))
        else:
            if self.model_type.lower() == 'snn':
                self.model.save(os.path.join(directory, self.name + 'e' + str(note) + '.pt'))
            elif self.model_type.lower() == 'ann':
                torch.save(self.model.state_dict(), os.path.join(directory, self.name + 'e' + str(note) + '.pt'))

        self.model.to(self.device)
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

    def set_policy(self, policy):
        self.policy = policy
        return None

    def learning(self, state):
        if not isinstance(state, bool):
            raise TypeError(state, ' must be a boolean variable.')

        if not self.model_type.lower() == 'snn':
            raise RuntimeError('Model in agent is not spike neural network, can not set learning state.')

        self.model.learning = state
        return None

    def make_action(self, observation):
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

        if self.policy == 'DDQN':
            action, action_index = self._decode_model_output(output, mode = 'mix')
        else:
            action, action_index = self._decode_model_output(output)
        return action, action_index, processed.cpu().detach(), output.cpu().detach()

    def random_action(self):
        return self.valid_action[random.randint(0, len(self.valid_action) - 1)]

    def insert_memory(self, observation):
        if isinstance(observation, torch.Tensor):
            observation = observation.numpy()

        observation = self.preprocess(observation, mode = 'init')
        self.memory = observation.to(self.device)
        return None

    def preprocess(self, observation, mode = 'normal'):
        if mode == 'normal':
            return self.transform(observation, self.memory)
        elif mode == 'init':
            return self.transform.insert_init_memory(observation)
        else:
            raise ValueError(mode, ' is invaled in PongAgent.preprocess().')

    def _decode_model_output(self, output, mode = 'sample', rand_p = None):
        if mode == 'argmax':
            _, action = torch.max(output, 1)
            action_index = action.cpu().detach().numpy()[0]
            action = self.action[action_index]
            return action, action_index
        elif mode == 'sample':
            try:
                output = output.detach().squeeze().cpu()
                m = Categorical(output)
                action_index = m.sample().numpy()
                action = self.valid_action[action_index]
                return action, action_index
            except RuntimeError:
                #one numbers in  probability distribution is zero
                _, action = torch.max(output, 0)
                action_index = action.cpu().detach().numpy()[0]
                action = self.valid_action[action_index]
                return action, action_index
        elif mode == 'mix':
            # rand_p is rnadom probability, if 1.0 means all action is random
            if rand_p is None:
                raise ValueError('Please set argument p in make_action')

            if random.random() < rand_p:
                #means random action
                action_index = random.randint(0, len(self.action) - 1)
                action = self.action[action_index]
                return action, action_index
            else:
                _, action = torch.max(output, 1)
                action_index = action.cpu().detach().numpy()[0]
                action = self.valid_action[action_index]
                return action, action_index

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


