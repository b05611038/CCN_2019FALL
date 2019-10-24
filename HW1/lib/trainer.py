import os
import time
import torch
import nengo
import numpy as np
import tensorflow as tf

import torch.cuda as cuda
import torch.nn as nn
from torch.optim import SGD, RMSprop, SGD
from torch.utils.data import DataLoader

from lib.utils import *

from lib.model.cnn import CNN
from lib.model.snn import SNN

from lib.data.mnist import MNIST
from lib.data.dataset import TorchMNIST, NengoMNIST
from lib.data.loader import NengoDLDataLoader


class TorchTrainer(object):
    def __init__(self, config):
        self.cfg = load_config(config)
        self.model_name = self.cfg['model_name']
        self.save_dir = self._init_directory(self.model_name)
        self.device = self._init_device(self.cfg['device'])
        self.model = CNN(
                self.cfg['class_num'],
                self.cfg['conv_layers']['layer_num'],
                self.cfg['conv_layers']['filter_num'],
                self.cfg['conv_layers']['filter_size']
                )

        self.model = self.model.to(self.device)
        self.recorder = Recorder(['train_loss', 'test_loss', 'test_accuracy'])

        print('TrochTrainer initialize done !!!')

    def train(self, config = None):
        start_time = time.time()
        if config is None:
            cfg = self.cfg

        train_set = TorchMNIST(cfg['data_path'], mode = 'train')
        test_set = TorchMNIST(cfg['data_path'], mode = 'test')
        optim = self._init_optimizer(self.model, cfg)
        epochs = cfg['epochs']
        for i in range(epochs):
            print('Start training epoch: ', i + 1, ' ...')
            self._epoch(self.model, optim, cfg, train_set, test_set)

        self.recorder.write(os.join.path(self.save_dir, cfg['model_name']))

        if cfg['save_weight']:
            self.save(self.model_name)

        print('All training process finish, cause %s seconds.' % (time.time() - start_time)) 
        return None

    def save(self, name):
        state = {}
        model = self.model.cpu()
        state['weight'] = model.state_dict()
        state['args'] = {
                'num_classes': model.num_classes,
                'num_layers': model.num_layers,
                'num_filters': model.num_filters,
                'kernel_sizes': model.kernel_sizes
        }

        save_object(os.path.join(self.save_dir, name), state)
        return None

    def _epoch(self, model, optim, cfg, train_set, test_set = None):
        batch_size = cfg['batch_size']

        dataloader = DataLoader(train_set, batch_size = batch_size, shuffle = True)
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.to(self.device)

        model = model.train()  
        train_loss = []
        for iter, (images, labels) in enumerate(dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            optim.zero_grad()

            output = model(images)

            loss = criterion(output, labels)
            loss.backward()

            optim.step()

            loss = loss.detach().cpu()
            train_loss.append(loss)

            if (iter + 1) % 5 == 0:
                print('Iter:', iter + 1, '| Loss:', loss.numpy())

        train_loss = float(torch.tensor(train_loss).mean().numpy())
        if test_set is not None:
            with torch.no_grad():
                print('\nCalculating testing set ...\n')

                dataloader =  DataLoader(test_set, batch_size = batch_size, shuffle = False)

                model = model.eval()
                test_loss, test_total, test_correct = [], [], 0
                for iter, (images, labels) in enumerate(dataloader):
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    output = model(images)

                    _, prediction = torch.max(output, 1)

                    test_total.append(images.size(0))
                    test_correct += (prediction == labels).sum().item()
                    test_loss.append(criterion(output, labels).detach())

            test_total = torch.tensor(test_total).float()
            test_loss = torch.tensor(test_loss).float()
            test_loss = float((torch.sum(torch.mul(test_loss, test_total.float())) / torch.sum(test_total)).detach().numpy())
            test_acc = float((100 * test_correct / torch.sum(test_total)).detach())

            self.recorder.insert((train_loss, test_loss, test_acc))
            print('Epoch summary:\nTrain loss: %.4f | Testing loss: %.4f | Testing Acc.: %.2f'
                    % (train_loss, test_loss, test_acc))
        else:
            self.recorder.insert(train_loss, 0.0, 0.0)
            print('Epoch summary:\nTrain loss: %.4f' % train_loss)

        return None

    def _init_optimizer(self, model, cfg):
        if cfg['optim']['name'].lower() == 'sgd':
            return SGD(model.parameters(), **cfg['optim']['args'])
        elif cfg['optim']['name'].lower() == 'rmsprop':
            return RMSprop(model.parameters(), **cfg['optim']['args'])
        elif cfg['optim']['name'].lower() == 'adam':
            return Adam(model.parameters(), **cfg['optim']['args'])
        else:
            raise ValueError('Optimizer:', cfg['optim']['name'], 'is not implemented in TorchTrainer.')

    def _init_device(self, device):
        # select training environment and device
        print('Init training device and environment ...')
        if device < 0:
            training_device = torch.device('cpu')
            print('Envirnment setting done, using device: cpu')
        else:
            torch.backends.cudnn.benchmark = True
            cuda.set_device(device)
            training_device = torch.device('cuda:' + str(device))
            print('Envirnment setting done, using device: cuda:' + str(device))

        return training_device

    def _init_directory(self, model_name):
        # information saving directory
        # save model checkpoint and episode history
        if not os.path.exists(model_name):
            os.makedirs(model_name)

        save_dir = model_name
        print('All object (model checkpoint, trainning history, ...) would save in', save_dir)
        return save_dir


class NengoDLTrainer(object):
    def __init__(self, config):
        self.cfg = load_config(config)
        self.model_name = self.cfg['model_name']
        self.save_dir = self._init_directory(self.model_name)
        self.model = SNN(
                self.cfg['class_num'],
                self.cfg['conv_layers']['layer_num'],
                self.cfg['conv_layers']['filter_num'],
                self.cfg['conv_layers']['filter_size'],
                n_steps = self.cfg['sample_time'],
                minibatch_size = self.cfg['minibatch_size']
                )

        self.recorder = Recorder(['train_loss', 'test_loss', 'test_accuracy'])

        print('NengoDLTrainer initialize done !!!')

    def train(self, config = None):
        start_time = time.time()
        if config is None:
            cfg = self.cfg

        train_set = NengoMNIST(cfg['data_path'], mode = 'train')
        test_set = NengoMNIST(cfg['data_path'], mode = 'test')
        interface = self.model.train_interface()
        optim = self._init_optimizer(cfg)
        epochs = cfg['epochs']
        for i in range(epochs):
            print('Start training epoch: ', i + 1, ' ...')
            interface = self._epoch(interface, optim, cfg, train_set, test_set)
            self.model.load_trained(interface)

        self.recorder.write(self.save_dir, cfg['model_name'])

        if cfg['save_weight']:
            self.save(self.model_name)

        print('All training process finish, cause %s seconds.' % (time.time() - start_time)) 
        return None

    def save(self, name):
        model = self.model
        args = {
                'num_classes': model.num_classes,
                'num_layers': model.num_layers,
                'num_filters': model.num_filters,
                'kernel_sizes': model.kernel_sizes,
                'n_steps': model.n_steps,
                'minibatch_size': model.minibatch_size
        }
        save_object(os.path.join(self.save_dir, name), args)
        model.sim.save_params(os.path.join(self.save_dir, name))
        return None

    def _epoch(self, interface, optim, cfg, train_set, test_set = None):
        minibatch_size = cfg['minibatch_size']
        dataloader = NengoDLDataLoader(train_set, batch_size = minibatch_size, shuffle = True)
        train_loss = []
        for i in range(dataloader.batch()):
            data = dataloader.load()
            data = self._probe_stimulate(data, interface, 'train')

            loss = self._get_loss(interface, data)
            train_loss.append(loss)

            interface['simulator'].train(data, optim, shuffle = False,
                    objective = {interface['output']: self._objective})

        #train_loss = np.mean(np.array(train_loss))
        if test_set is not None:
            print('\nCalculating testing set ...\n')

            dataloader = NengoDLDataLoader(test_set, batch_size = minibatch_size, shuffle = False)

            test_loss, test_total, test_acc = [], [], []
            for i in range(dataloader.batch()):
                data = dataloader.load()
                test_total.append(data[0].shape[0])

                loss_data = self._probe_stimulate(data, interface, 'test_loss')
                val_loss = self._get_loss(interface, loss_data)
                test_loss.append(val_loss)

                acc_data = self._probe_stimulate(data, interface, 'test_acc')
                val_acc = self._get_acc(interface, acc_data)
                test_acc.append(val_acc)

            test_total = np.array(test_total)
            test_loss = float(np.sum(np.array(test_loss) * test_total) / np.sum(test_total))
            test_acc = float(np.sum(np.array(test_acc) * test_total) / np.sum(test_total))

            self.recorder.insert((train_loss, test_loss, test_acc))
            print('Epoch summary:\nTrain loss: %.4f | Testing loss: %.4f | Testing Acc.: %.2f'
                    % (train_loss, test_loss, test_acc))

        else:
            self.recorder.insert(train_loss, 0.0, 0.0)
            print('Epoch summary:\nTrain loss: %.4f' % train_loss)

        return interface

    def _probe_stimulate(self, data, interface, mode):
        images, labels = data
        images = images.reshape(images.shape[0], -1)
        if mode == 'train':
            data = {interface['input']: images[:, None, :],
                    interface['output']: labels[:, None, :]}
            return data
        if mode == 'test_loss':
            data = {interface['input']: images[:, None, :],
                    interface['output']: labels[:, None, :]}
            return data
        elif mode == 'test_acc':
            data = {interface['input']: images[:, None, :],
                    interface['output_filter']: labels[:, None, :]}
            return data
        else:
            raise ValueError(mode, 'is not a valid slelection for time sampler')

    def _objective(self, outputs, targets):
        return tf.nn.softmax_cross_entropy_with_logits_v2(logits = outputs, labels = targets)

    def _get_loss(self, interface, data_pair):
        def error_func(outputs, targets):
            error = -tf.reduce_mean(tf.reshape(targets, (-1, self.cfg['class_num'])) * \
                    tf.log(tf.clip_by_value(tf.reshape(outputs, (-1, self.cfg['class_num'])), 1e-7, 1.0)))
            return error

        loss = interface['simulator'].loss(data_pair, {interface['output']: error_func})
        return loss
 
    def _get_acc(self, interface, data_pair):
        def error_func(outputs, targets):
            error = tf.reduce_mean(tf.cast(tf.not_equal(tf.argmax(outputs[:, -1], axis= -1),
                    tf.argmax(targets[:, -1], axis = -1)), tf.float32))
            return error
 
        error = interface['simulator'].loss(data_pair, {interface['output_filter']: error_func})
        return 100 * (1 - error)

    def _init_optimizer(self, cfg):
        if cfg['optim']['name'].lower() == 'sgd':
            return tf.train.MomentumOptimizer(**cfg['optim']['args'])
        elif cfg['optim']['name'].lower() == 'rmsprop':
            return RMSpropOptimizer(**cfg['optim']['args'])
        elif cfg['optim']['name'].lower() == 'adam':
            return AdamOptimizer(**cfg['optim']['args'])
        else:
            raise ValueError('Optimizer:', cfg['optim']['name'], 'is not implemented in NengoDLTrainer.')

    def _init_directory(self, model_name):
        # information saving directory
        # save model checkpoint and episode history
        if not os.path.exists(model_name):
            os.makedirs(model_name)

        save_dir = model_name
        print('All object (model checkpoint, trainning history, ...) would save in', save_dir)
        return save_dir


