import os
import nengo
import nengo_dl
import numpy as np
import tensorflow as tf 

import torch
import torchvision
import torch.cuda as cuda
import torchvision.transforms as tfs 

from PIL import Image
from torch.utils.data import DataLoader

import lib.model

from lib.utils import *

from lib.data.dataset import TorchInference

from lib.model.cnn import CNN
from lib.model.snn import SNN


class Container(object):
    # class to intergrate pytorch model and nengo model for classification problems
    def __init__(self, model_path, model_type):
        if not os.path.isdir(model_path):
            raise RuntimeError('Can not detect model object, please check your file path.')

        self.model_name = model_path
        self.model_path = os.path.join(model_path, model_path)
        if model_type not in ['torch', 'nengo']:
            raise ValueError('Container object only support for pytorch model or nengo model.')

        self.model_type = model_type
        self.model = self._load_model(model_path, model_type)
        self.nengo_reload = False

    def inference(self, image, mode = 'numpy', transforms = None):
        if mode not in ['numpy', 'torch']:
            raise ValueError('lib.model.Container only support inference in numpy mode or torch mode.')

        if self.model_type == 'torch':
            outcome = self._torch_inference(self.model, image, transforms)
            outcome = outcome.detach().cpu()
        elif self.model_type == 'nengo':
            image = np.array(image)
            if self.nengo_reload:
                self._load_model(self.model_name, self.model_type)

            outcome = self.model(image)
            self.nengo_reload = True

        if mode == 'numpy':
            outcome = np.array(outcome)
        elif mode == 'torch':
            outcome = torch.tensor(outcome)

        return outcome

    def _torch_inference(self, model, data, transforms):
        dataset = TorchInference(data, transforms)
        dataloader = DataLoader(dataset, batch_size = len(dataset), shuffle = False)
        for iter, (data) in enumerate(dataloader):
            data = data.to(self.device)

            outcome = self.model(data)
            outcome = outcome.cpu()

        return outcome

    def _load_model(self, save_dir, model_type):
        state = load_pickle_obj(os.path.join(save_dir, save_dir + '.pkl'))
        if model_type == 'torch':
            model = CNN(
                    state['args']['num_classes'],
                    state['args']['num_layers'],
                    state['args']['num_filters'],
                    state['args']['kernel_sizes']
                    )
            model.load_state_dict(state['weight'])
            self.device = self._init_torch_device()
            model = model.eval()
            model = model.to(self.device)
        elif model_type == 'nengo':
            model = SNN(
                    state['num_classes'],
                    state['num_layers'],
                    state['num_filters'],
                    state['kernel_sizes'],
                    n_steps = state['n_steps'],
                    minibatch_size = state['minibatch_size']
                    )

            model.sim.load_params(os.path.join(save_dir, save_dir))

        return model

    def _init_torch_device(self):
        # select training environment and device
        print('Init training device and environment ...')
        if cuda.is_available():
            torch.backends.cudnn.benchmark = True
            cuda.set_device(0)
            training_device = torch.device('cuda:' + str(0))
            print('Envirnment setting done, using device: cuda:' + str(0))
        else:
            training_device = torch.device('cpu')
            print('Envirnment setting done, using device: cpu')

        return training_device

