import os
import time
import torch
import nengo
import numpy as np
import torch.cuda as cuda
import tensorflow as tf

from torch.optim import SGD, RMSprop, SGD
from torch.utils.data import DataLoader

from lib.utils import *
from lib.data.mnist import MNIST
from lib.data.dataset import TorchMNIST 
from lib.model.cnn import CNN


class TorchTrainer(object):
    def __init__(self, config):
        self.cfg = load_config(config)
        self.model_name = self.cfg['model_name']
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
            self._epoch(model, optim, cfg, train_set, test_set)

        self.recorder.write('./', cfg['model_name'])

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

        save_object(self.cfg['model_name'], state)
        return None

    def _epoch(self, model, optim, cfg, train_set, test_set = None):
        batch_size = cfg['batch_size']

        dataloader = DataLoader(train_set, batch_size = batch_size, shuffle = True)
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.to(self.device)

        model = model.train()  
        train_loss = []
        for iter, (image, label) in enumerate(dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            optim.zero_grad()

            output = model(images)

            loss = criterion(output, labels)
            loss.backward()

            optim.step()

            loss = loss.detach().cpu()
            total_loss.append(loss)

            print('Iter:', iter, '| Loss:', loss.numpy())

        train_loss = float(torch.tensor(train_loss).mean().numpy())
        if test_set is not None:
            with torch.zero_grad():
                print('\nCalculating testing set ...\n')

                dataloader =  DataLoader(test_set, batch_size = batch_size, shuffle = False)

                model = model.eval()
                test_loss, test_total, test_correct = [], [], 0
                for iter, (image, label) in enumerate(dataloader):
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    output = model(images)

                    _, prediction = torch.max(output, 1)

                    test_total.append(images.size(0))
                    test_correct += (prediction == labels).sum().item()
                    test_loss.append(criterion(output, labels).detach())

            test_loss = float((torch.sum(torch.mul(test_loss, test_total.float())) / torch.sum(test_total)).detach().numpy())
            test_acc = float((100 * test_correct / torch.sum(test_total)).detach())

            self.recorder.insert(train_loss, test_loss, test_acc)
            print('Epoch summary:\nTrain loss: %.4f | Testing loss: %.4f | Testing Acc.: %.2f'
                    % (train_loss, test_loss, test_acc))
        else:
            self.recorder.insert(train_loss, 0.0, 0.0)
            print('Epoch summary:\nTrain loss: %.4f' % train_loss)

        return None

    def _init_optimizer(self, model, cfg):
        if cfg['optim']['name'].lower() == 'sgd':
            return SGD(model.parameters(), **cfg['args'])
        elif cfg['optim']['name'].lower() == 'rmsprop':
            return RMSprop(model.parameters(), **cfg['args'])
        elif cfg['optim']['name'].lower() == 'adam':
            return Adam(model.parameters(), **cfg['args'])
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


class NengoTrainer(object):
    def __init__(self, config):
        self.cfg = config


