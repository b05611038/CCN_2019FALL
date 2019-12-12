import os
import sys
import copy
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
from lib.agent import load_agent, PongAgent
from lib.environment import GymEnvironment


__all__ = ['select_trainer','ANNBaseTrainer', 'QTrainer', 'PGTrainer']


def select_trainer(config):
    policy_list = ['PO', 'PPO', 'DDQN', 'A2C']
    cfg = load_config(config)
    if cfg['policy'] == 'DDQN':
        return QTrainer(config)
    elif cfg['policy'] == 'PO' or cfg['policy'] == 'PPO' or cfg['policy'] == 'A2C':
        return PGTrainer(config)
    else:
         raise ValueError(cfg['policy'], 'is not in ' + str(policy_list))

class ANNBaseTrainer(object):
    def __init__(self, config):
        self.cfg = load_config(config)
        self.model_name = self.cfg['model_name']
        self.save_dir = self._init_directory(self.model_name)
        self.device = self._init_device(self.cfg['device'])
        self.policy = self.cfg['policy']
        self.env = self._init_env(self.cfg['env'])
        self.agent, self.state = self._init_agent(self.cfg['agent'], self.env, self.name)
        self.agent.set_policy(self.policy)
        self.model = self.agent.model
        self._init_loss_layer(self.policy)
        self.optim = self._init_optimizer(self.model, self.cfg['optim'])
        self.reward_preprocess = self.cfg['dataset']['reward_preprocess']
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

    def _collect_data(self):
        raise NotImplementedError()

    def _update_policy(self):
        raise NotImplementedError()

    def _calculate_loss(self):
        raise NotImplementedError()

    def _init_loss_layer(self):
        raise NotImplementedError()

    def _init_agent(self, config, env, name):
        if config['model_type'].lower() != 'ann':
            raise RuntimeError('Only "ann" base agent can be trained in ANNTrainer.')

        if os.path.isfile(os.path.join(self.save_dir, 'agent.pkl')):
            folder = os.path.join()

            agent = load_agent(os.path.join(self.save_dir, 'agent.pkl'),
                    os.path.join(self.save_dir, self.model_name + '.pt'))

            if agent.note is not None:
                state = agent.note
            else:
                state = 0
        else:
            agent = PongAgent(
                    name,
                    config['model_type'],
                    config['model_config'],
                    config['preprocess'],
                    self.device
                    )

            state = 0

        return agent, state

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


class QTrainer(ANNBaseTrainer):
    def __init__(self, config):
        super(QTrainer, self).__init__(config):
        if self.continue_training:
            self.random_probability = 0.025
        else:
            self.random_probability = 1.0

        self.target_net = copy.deepcopy(self.model)
        self.gamma = 0.99
        self.dataset = QReplayBuffer(env = env, maximum = self.cfg['dataset']['maximum'],
                preprocess_dict = reward_preprocess)

    def play(self, config = None):
        if config is None:
            cfg = self.cfg

        max_state = cfg['episodes']
        episode_size = cfg['dataset']['maximum']
        batch_size = cfg['dataset']['batch_size']
        save_interval = cfg['checkpoint']

        if self.random_action:
            self.decay = (1.0 - 0.025) / max_state

        state = self.state
        max_state += self.state
        record_round = (max_state - state) // 100

        while(state < max_state):
            start_time = time.time()
            self.agent.model = self.model
            reward, reward_mean = self._collect_data(self.agent, episode_size, mode = 'train')
            if self.dataset.trainable():
                loss = self._update_policy(batch_size)

                if state % record_round == record_round - 1:
                    _, test_reward = self._collect_data(self.agent, 10, mode = 'test')
                    fix_reward = self._fix_game(self.agent)
                    self.recorder.insert([state, loss, reward_mean, test_reward])
                    print('\nTraing state:', state + 1, '| Loss:', loss, '| Mean reward:', reward_mean,
                            '| Test game reward:', test_reward, '| Fix game reward:', fix_reward,
                            '| Spend Times: %.4f seconds' % (time.time() - start_time), '\n')
                else:
                    self.recorder.insert([state, loss, reward_mean, 'NaN'])
                    print('Traing state:', state + 1, '| Loss:', loss, '| Mean reward:', reward_mean,
                            '| Spend Times: %.4f seconds' % (time.time() - start_time), '\n')

                state += 1
            else:
                continue

            if self.random_action:
                self._adjust_probability()

            if state % save_interval == 0 and state != 0:
                self.save(episode_note = state)

        self.save(episode_note = state)
        self.recorder.write(self.save_dir, 'his_' + self.model_name + '_s' + str(self.state) + '_s' + str(max_state))
        print('Training Agent:', self.model_name, 'finish.')
        return None

    def _collect_data(self, agent, rounds, mode = 'train'):
        agent.model = self.model
        print('Start interact with environment ...')
        final_reward = []
        for i in range(rounds):
            done = False
            true_done = False
            skip_first = True
            _ = self.env.reset()

            mini_counter = 0
            final_reward.append(0.0)
            last_observation = None
            last_action = None
            last_reward = None
            while not true_done:
                if skip_first:
                    observation, _r, _d, _ = self.env.step(agent.init_action())
                    agent.insert_memory(observation)
                    skip_first = False
                    continue

                action, action_index, processed, model_out = agent.make_action(observation,
                        p = self.random_probability)

                observation_next, reward, done, _ = self.env.step(action)
                final_reward[i] += reward

                if mode == 'train' and last_observation is not None:
                    if reward == 0.0:
                        self.dataset.insert(last_observation, processed, last_action)
                        mini_counter += 1
                    else:
                        self.dataset.insert(last_observation, processed, last_action)
                        mini_counter += 1
                        self.dataset.insert_reward(reward, mini_counter)
                        mini_counter = 0

                    if done or true_done:
                        self.dataset.insert_reward(-1.0, mini_counter)
                        mini_counter = 0
                        skip_first = True
                        last_observation = None
                        last_action = None
                        last_reward = None

                elif mode == 'test':
                    pass

                last_observation = processed
                last_action = action_index
                last_reward = reward

                observation = observation_next

            if i % 5 == 4:
                print('Progress:', i + 1, '/', rounds)

        final_reward = np.array(final_reward)
        reward_mean = np.mean(final_reward)
        print('Data collecting process finish.')

        return final_reward, reward_mean

    def _update_policy(self, batch_size):
        self.model = self.model.train().to(self.device)
        final_loss = []
        loader = DataLoader(self.dataset, batch_size = batch_size, shuffle = False)
        for iter, (observation, next_observation, action, reward) in enumerate(loader):
            observation = observation.to(self.device)
            observation_next = next_observation.to(self.device)
            action = action.to(self.device)
            reward = reward.to(self.device)
            
            self.optim.zero_grad()
            
            loss = self._calculate_loss(observation, observation_next, action, reward, self.model, self.target_net)
            loss.backward()

            for param in self.model.parameters():
                param.grad.data.clamp_(-1, 1)

            self.optim.step()

            final_loss.append(loss.detach().cpu())

            print('Mini batch progress:', iter + 1, '| Loss:', loss.detach().cpu().numpy())

        self.target_net = copy.deepcopy(self.policy_net)
        self.target_net = self.target_net.eval()

        final_loss = torch.mean(torch.tensor(final_loss)).detach().numpy()

        return final_loss

    def _adjust_probability(self):
        self.random_probability -= self.decay
        return None

    def _one_hot(self, length, index):
        return torch.index_select(torch.eye(length), dim = 0, index = index.cpu())

    def _calculate_loss(self, observation, next_observation, action, reward, policy_net, target_net):
        mask = self._one_hot(len(self.valid_action), action)
        mask = mask.byte().to(self.device)

        last_output = policy_net(observation)
        state_action_values = torch.masked_select(last_output, mask = mask)
        next_state_values, _ = torch.max(target_net(next_observation), 1)
        next_state_values = next_state_values.detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward
        loss = self.loss_layer(state_action_values, expected_state_action_values)
        return loss

    def _init_loss_layer(self, policy):
        self.loss_layer = nn.L1Loss()
        return None


class PGTrainer(ANNBaseTrainer):
    def __init__(self, config):
        super(PGTrainer, self).__init__(config):

        self.eps = 10e-7
        self.dataset = PGReplayBuffer(env = env, maximum = self.cfg['dataset']['maximum'],
                    preprocess_dict = reward_preprocess)

    def play(self, config):
        if config is None:
            cfg = self.cfg

        max_state = cfg['episodes']
        episode_size = cfg['dataset']['maximum']
        batch_size = cfg['dataset']['batch_size']
        save_interval = cfg['checkpoint']

        state = self.state
        max_state += self.state
        record_round = (max_state - state) // 100

        while(state < max_state):
            start_time = time.time()
            self.agent.model = self.model
            reward, reward_mean = self._collect_data(self.agent, episode_size, mode = 'train')
            if self.dataset.trainable():
                loss = self._update_policy(episode_size, batch_size)

                if state % record_round == record_round - 1:
                    _, test_reward = self._collect_data(self.agent, 10, mode = 'test')
                    fix_reward = self._fix_game(self.agent)
                    self.recorder.insert([state, loss, reward_mean, test_reward, fix_reward])
                    print('\nTraing state:', state + 1, '| Loss:', loss, '| Mean reward:', reward_mean,
                            '| Test game reward:', test_reward, '| Fix game reward:', fix_reward,
                            '| Spend Times: %.4f seconds' % (time.time() - start_time), '\n')
                else:
                    self.recorder.insert([state, loss, reward_mean, 'NaN', 'NaN'])
                    print('Traing state:', state + 1, '| Loss:', loss, '| Mean reward:', reward_mean,
                            '| Spend Times: %.4f seconds' % (time.time() - start_time), '\n')

                state += 1
            else:
                continue

            if state % save_interval == 0 and state != 0:
                self._save_checkpoint(state)

        self._save_checkpoint(state)
        self.recorder.write(self.save_dir, 'his_' + self.model_name + '_s' + str(self.state) + '_s' + str(max_state))
        print('Training Agent:', self.model_name, 'finish.')
        return None

    def _collect_data(self, agent, rounds, mode = 'train'):
        agent.model = self.model
        print('Start interact with environment ...')
        final_reward = []
        for i in range(rounds):
            done = False
            skip_first = True
            _ = self.env.reset()

            if mode == 'train':
                self.dataset.new_episode()

            time_step = 0
            mini_counter = 0
            final_reward.append(0.0)
            while not done:
                if skip_first:
                    observation, _r, _d, _ = self.env.step(agent.random_action())
                    agent.insert_memory(observation)
                    skip_first = False
                    continue

                action, action_index, processed, model_out = agent.make_action(observation)
                observation_next, reward, done, _ = self.env.step(action)
                final_reward[i] += reward

                if mode == 'train':
                    if reward == 0:
                        self.dataset.insert(processed, model_out)
                        mini_counter += 1
                    else:
                        self.dataset.insert(processed, model_out)
                        mini_counter += 1
                        self.dataset.insert_reward(reward, mini_counter, done)
                        mini_counter = 0

                elif mode == 'test':
                    pass

                observation = observation_next
                time_step += 1

            if i % 5 == 4:
                print('Progress:', i + 1, '/', rounds)

        final_reward = np.asarray(final_reward)
        reward_mean = np.mean(final_reward)
        print('Data collecting process finish.')

        return final_reward, reward_mean

    def _update_policy(self, episode_size, batch_size, times = 5):
        self.model = self.model.train().to(self.device)
        final_loss = []
        for iter in range(times):
            if self.policy == 'A2C':
                self.dataset.make(episode_size)
            else:
                self.dataset.make(episode_size * 2)

            loss_temp = []
            loader = DataLoader(self.dataset, batch_size = batch_size, shuffle = False)
            for mini_iter, (observation, action, reward) in enumerate(loader):
                observation = observation.to(self.device)
                action = action.to(self.device)
                reward = reward.to(self.device)
                
                self.optim.zero_grad()
                
                output, state_value = self.model(observation)
                
                loss = self._calculate_loss(output, action, state_value, reward)
                loss.backward()
                
                self.optim.step()

                loss_temp.append(loss.detach().cpu())

            loss = torch.mean(torch.tensor(loss_temp))
            final_loss.append(loss.detach().cpu())

            if times != 1:
                print('Mini batch progress:', iter + 1, '| Loss:', loss.detach().cpu().numpy())

        final_loss = torch.mean(torch.tensor(final_loss)).detach().numpy()

        return final_loss

    def _calculate_loss(self):
        _, target = torch.max(record, 1)
        target = target.detach()

        if self.policy == 'PO':
            action = torch.log(action + self.eps)
            loss = self.loss_layer(action, target)
            loss = torch.mean(loss * reward, dim = 0)
            return loss
        elif self.polict == 'PPO':
            important_weight = self._important_weight(record, action, target)
            important_weight = torch.clamp(important_weight, 1.0 - self.clip_value, 1.0 + self.clip_value)
            action = torch.log(action + self.eps)
            loss = self.loss_layer(action, target)
            loss = torch.mean(loss * reward * important_weight, dim = 0)
            return loss

        elif self.policy == 'A2C':
            value_loss = self.critic_loss(state_value, reward)
            advantage = reward - state_value.squeeze().detach()
            action = torch.log(action + self.eps)
            loss = self.loss_layer(action, target)
            loss = torch.mean(loss * advantage, dim = 0)
            return loss + value_loss

    def _important_weight(self, record, action, target):
        important_weight = action / record + self.eps
        target = target.repeat([2, 1]).transpose(0, 1)
        important_weight = torch.gather(important_weight, 1, target)
        important_weight = torch.mean(important_weight, dim = 1)
        return important_weight.detach()

    def _init_loss_layer(self, policy):
        self.loss_layer = nn.NLLLoss(reduction = 'none')
        if policy == 'A2C'
            self.critic_loss = nn.L1Loss()

        if policy == 'PPO':
            self.clip_value = 0.2

        return None


